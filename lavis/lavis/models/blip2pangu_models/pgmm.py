import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch_npu  

from torch.npu.amp import autocast, GradScaler
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train

from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url

from lavis.models.blip2pangu_models.tokenization_openpangu import PanguTokenizer
from lavis.models.blip2pangu_models.configuration_openpangu_dense import PanguEmbeddedConfig
from lavis.models.blip2pangu_models.modeling_openpangu_dense import PanguEmbeddedForCausalLM

from lavis.models.blip2pangu_models.pgmm_qformer import Blip2QformerForPanGuMM
from lavis.models.blip2pangu_models.projector import build_vision_projector

from lavis.models.blip2pangu_models.pgmmconfig import PGNMultiModalConfig, VisionProjectorConfig

@registry.register_model("blip2_pangu")

class Blip2PanGu(Blip2Base):
    """
    BLIP2 PanGu model.
    Supported model types:
        - pretrained_pangu7b: pretrained model with PanGu7b

    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_pangu": "configs/models/blip2/blip2_pretrain_openpangu7b.yaml", 
    }

    """
    Config:
        1.self.config == self.llm.config 盘古的config
        2.为了方便,我把一些配置从"pretrain_pangu"搬到了 self.mm_config == mm_config == PGNMultiModalConfig()中
    """
    def __init__(
        self,
        prompt = "",
        max_txt_len = 32,
        llm_hidden_size = 4096,
        device = None,
        mm_config = None,
        dtype = None
    ):
        super().__init__()
        if mm_config is None:
            mm_config = PGNMultiModalConfig()
        self.mm_config = mm_config # 整个多模态模型的配置，由我们自主定义
        
        llm_kwargs: Dict[str, Any] = {}
        if dtype is None:
            dtype = self.mm_config.dtype
            llm_kwargs["torch_dtype"] = dtype  
            self.dtype = dtype
        else:
            llm_kwargs["torch_dtype"] = dtype  
            self.dtype = dtype

        self.llm: PanguEmbeddedForCausalLM = PanguEmbeddedForCausalLM.from_pretrained(
            mm_config.llm_path, **llm_kwargs
        )
        if device is not None:
            self.llm.to(device)

        self.llm_config: PanguEmbeddedConfig = self.llm.config

        vocab_file = os.path.join(mm_config.llm_path, "tokenizer.model")
        if not os.path.isfile(vocab_file):
            raise FileNotFoundError(
                f"Could not find tokenizer.model at '{vocab_file}'. "
                "Make sure `llm_path` points to the OpenPangu checkpoint directory."
            )
        self.tokenizer = PanguTokenizer(vocab_file=vocab_file) # 加载盘古tokenizer 

        for name, param in self.llm.named_parameters():
            param.requires_grad = False       # 1 多模态流程中盘古冻结不参与训练 2 如需训练，微调不应选择全量微调，应使用PEFT

        print("Checking LLM freeze status:")

        for name, param in self.llm.named_parameters():
            if param.requires_grad:
                print(f"Warning: {name} is NOT frozen!")
                break # 只打印第一个看看就行

        # #adapter微调用，确保只有adapter层的参数是可训练的
        # for name, param in self.llm.named_parameters():
        #     if "adapter" in name:  # 假设 Adapter 层的参数名中包含 "adapter"
        #         param.requires_grad = True
        #         #print(f"\n-------------Warning: {name} is NOT frozen!-------------")
        #         # print("[INFO] Adapter parameters are trainable.")
        #     else:
        #         param.requires_grad = False
        
                

        self.vision_frontend = Blip2QformerForPanGuMM( 
            img_size=mm_config.image_size,
            drop_path_rate=0.0,
            use_grad_checkpoint=False,
            vit_precision=mm_config.vit_precision,
            freeze_vit=mm_config.freeze_vision,
            num_query_token=mm_config.num_query_token,
        )
        if device is not None:
            self.vision_frontend.to(device)

        if llm_hidden_size is not None:
            proj_cfg = VisionProjectorConfig()
            self.pangu_proj = build_vision_projector(proj_cfg)
        else:
            raise ValueError('need projector!')
        
        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
 
    @property
    def config(self):
        return self.llm_config
        
    def forward(self, samples):
        image = samples["image"]
        q_hidden = self.vision_frontend(image)

        inputs_pg = self.pangu_proj(q_hidden)
        atts_pg = torch.ones(inputs_pg.size()[:-1], dtype=torch.long).to(image.device) # [batch_size,num_query_token] 大小的，全1 的注意力张量

        self.tokenizer.padding_side = "right"  # 其实没必要，盘古默认的paddingside就是右边

        text = [t  for t in samples["text_input"]] # 这种组织方式似乎与训练数据集数据组织方式有关

        # 注意此处与blip2_opt的逻辑是不同的
        if self.tokenizer.eos_token is None:
            eos_token_str = self.tokenizer.decode(self.tokenizer.eos_token_id)
        else:
            eos_token_str = self.tokenizer.eos_token
        text = [t + eos_token_str for t in text]

        pg_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = pg_tokens.input_ids.masked_fill(pg_tokens.input_ids == self.tokenizer.pad_token_id, -100)  # 填充padding的label为-100
        if self.prompt:  # 如果有prompt prompt也不参与计算loss
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (torch.ones(atts_pg.size(), dtype=torch.long).to(image.device).fill_(-100))
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm.model.embed_tokens(pg_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_pg, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_pg, pg_tokens.attention_mask], dim=1)   # 拼接方式是将所有的图像放前面，把文本信息放在后面

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        with torch.npu.amp.autocast( 
            enabled = False
            # enabled = True 推理打开
 
        ):          
            
            q_hidden = self.vision_frontend(image)
            inputs_pg = self.pangu_proj(q_hidden)
            atts_pg = torch.ones(inputs_pg.size()[:-1], dtype=torch.long).to(image.device)  # [batch_size, query_num] 形状的张量，全1 ，要参与计算

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0) # 复制prompt到batch_size份

            pg_tokens = self.tokenizer(prompt, return_tensors="pt").to(image.device)
            inputs_embeds = self.llm.model.embed_tokens(pg_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_pg, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_pg, pg_tokens.attention_mask], dim=1) 

            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            prompt_length = pg_tokens.input_ids.shape[1]
            output_text = self.tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text
        
    @classmethod
    def from_config(cls, cfg):

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            prompt=prompt,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def load_from_pretrained(self, url_or_filename):
        logging.info("[Blip2pangu]: Start to load pretrained model from %s" % url_or_filename)
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        # Remap keys for vision_frontend (for loading Qformer weights from BLIP2 checkpoints)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("visual_encoder") or key.startswith("ln_vision") or key.startswith("Qformer") or key.startswith("query_tokens"):
                new_key = "vision_frontend." + key
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value  # Keep other keys unchanged
        state_dict = new_state_dict

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("Unexpected keys %s", msg.unexpected_keys)
        logging.info("Loaded checkpoint from %s" % url_or_filename)

        return msg


