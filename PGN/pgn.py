import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch_npu  

from torch.npu.amp import autocast, GradScaler
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# from q_former.registry import registry
from q_former.blip2 import Blip2Base, disabled_train

from q_former.dist_utils import download_cached_file
from q_former.utils import is_url

from llm.tokenization_openpangu import PanguTokenizer
from llm.configuration_openpangu_dense import PanguEmbeddedConfig
from llm.modeling_openpangu_dense import PanguEmbeddedForCausalLM

from q_former.q_former import Blip2QformerForPanGuMM
from q_former.projector import build_vision_projector

from q_former.pgnconfig import PGNMultiModalConfig, VisionProjectorConfig

# @registry.register_model("blip2_pangu")
IGNORE_INDEX = -100


class Blip2PanGu(Blip2Base):
    """
    BLIP2 PanGu model.
    Supported model types:
        - pretrained_pangu7b: pretrained model with PanGu7b

    """
    """
    Config:
        1.self.config == self.llm.config 盘古的config
        2.为了方便,我把一些配置从"pretrain_pangu"搬到了 self.mm_config == mm_config == PGNMultiModalConfig()中
    """
    def __init__(
        self,
        prompt0 = "You are a helpful navigation assistant. Based on what you see, please output the action you should take. Your output should only include: move forward, turn left 15 degrees, turn right 15 degrees, or stop.",
        prompt2 = "These are the scenes you saw and the actions you took:",
        prompt1 = "The last picture you see is what you see from your current location.",
        prompt_action = "action taken:",
        prompt_answer = "please take your action now:",
        max_txt_len = 4096,
        llm_hidden_size = 4096,
        device = None,
        mm_config = None,
        dtype = None
    ):
        super().__init__()
        target_device = device
        

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

        if mm_config.load_llm_pretrained:
            self.llm: PanguEmbeddedForCausalLM = PanguEmbeddedForCausalLM.from_pretrained(
                mm_config.llm_path, **llm_kwargs
            )
        else:
            llm_config = PanguEmbeddedConfig.from_pretrained(mm_config.llm_path)
            self.llm = PanguEmbeddedForCausalLM(llm_config)
            if dtype is not None:
                self.llm.to(dtype=dtype)

        self.llm_config: PanguEmbeddedConfig = self.llm.config

        vocab_file = os.path.join(mm_config.llm_path, "tokenizer.model")
        if not os.path.isfile(vocab_file):
            raise FileNotFoundError(
                f"Could not find tokenizer.model at '{vocab_file}'. "
                "Make sure `llm_path` points to the OpenPangu checkpoint directory."
            )
        self.tokenizer = PanguTokenizer(vocab_file=vocab_file) # 加载盘古tokenizer 

        for name, param in self.llm.named_parameters():
            param.requires_grad = False       # PEFT微调盘古

        print("Checking LLM freeze status:")
        for name, param in self.llm.named_parameters():
            if param.requires_grad:
                print(f"Warning: {name} is NOT frozen!")
                # break # 只打印第一个看看就行

        self.vision_frontend = Blip2QformerForPanGuMM( 
            img_size=mm_config.image_size,
            drop_path_rate=0.0,
            use_grad_checkpoint=False,
            vit_precision=mm_config.vit_precision,
            freeze_vit=mm_config.freeze_vision,
            num_query_token=mm_config.num_query_token,
            load_vision_pretrained=mm_config.load_vision_pretrained,
            load_qformer_pretrained=mm_config.load_qformer_pretrained,
        )

        if llm_hidden_size is not None:
            proj_cfg = VisionProjectorConfig()
            self.pangu_proj = build_vision_projector(proj_cfg)
        else:
            raise ValueError('need projector!')

        self.max_txt_len = max_txt_len
        self.prompt0 = prompt0
        self.prompt1 = prompt1
        self.prompt2 = prompt2
        self.prompt_action = prompt_action
        self.prompt_answer = prompt_answer

        new_special_tokens = [
            "<|image_start|>", 
            "<|image_end|>", 
            "<|frame_sep|>" 
        ]

        num_new_tokens = len(new_special_tokens)

        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': new_special_tokens}
        )

        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.requires_grad_(False)

        input_embeddings = self.llm.get_input_embeddings().weight.data
        output_embeddings = self.llm.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        if target_device is not None:
            self.llm.to(target_device)
            self.vision_frontend.to(target_device)
            self.pangu_proj.to(target_device)
 
    @property
    def config(self):
        return self.llm_config

    def _trim_history_actions(
        self,
        question: str,
        answer: str,
        history_actions: List[str],
        num_images: int,
        eos_token_str: str,
    ) -> Tuple[List[str], int, int]:
        if not history_actions or not self.max_txt_len:
            return history_actions, num_images, 0

        image_start = "<|image_start|>"
        image_end = "<|image_end|>"
        frame_sep = "<|frame_sep|>"

        prefix = f"{self.prompt0}\n{question}\n{self.prompt2}\n{image_start}"
        suffix = f"{image_end}\n{self.prompt1}\n{image_start}{image_end}\n{self.prompt_answer}\n{answer}{eos_token_str}"

        prefix_len = len(self.tokenizer.encode(prefix, add_special_tokens=False))
        suffix_len = len(self.tokenizer.encode(suffix, add_special_tokens=False))
        sep_len = len(self.tokenizer.encode(frame_sep, add_special_tokens=False))
        bos_len = 1 if getattr(self.tokenizer, "add_bos_token", False) else 0

        action_chunks = [f"{self.prompt_action}{action}" for action in history_actions]
        action_lens = [len(self.tokenizer.encode(chunk, add_special_tokens=False)) for chunk in action_chunks]

        safety_margin = 8
        max_hist_tokens = self.max_txt_len - (prefix_len + suffix_len + bos_len + safety_margin)
        if max_hist_tokens <= 0:
            keep_actions = 1
        else:
            total = 0
            keep_actions = 0
            for length in reversed(action_lens):
                add = length + (sep_len if keep_actions > 0 else 0)
                if total + add > max_hist_tokens:
                    break
                total += add
                keep_actions += 1
            if keep_actions == 0:
                keep_actions = 1

        if keep_actions >= len(history_actions):
            return history_actions, num_images, 0

        trimmed_actions = history_actions[-keep_actions:]
        keep_num_images = min(num_images, keep_actions + 1)
        if keep_num_images < 2:
            return history_actions, num_images, 0

        keep_actions = min(keep_actions, keep_num_images - 1)
        trimmed_actions = trimmed_actions[-keep_actions:]
        image_start_idx = max(0, num_images - (keep_actions + 1))
        return trimmed_actions, keep_actions + 1, image_start_idx
        
    def forward(self, samples,):

        image = samples["images"]
        num_images = samples["num_images"]
        b, n, c, h, w = image.shape  # n是整个批处理中所有样本中的最大样本（最大图像数），不足的样本以黑图padding
        image_reshaped = image.view(b * n, c, h, w)
        q_hidden = self.vision_frontend(image_reshaped)
        inputs_pg = self.pangu_proj(q_hidden)
        img_seq_len = inputs_pg.shape[1] 
        img_dim = inputs_pg.shape[2]
        # inputs_pg.shape == [b*n, tokens_per_img, dim]
        inputs_pg = inputs_pg.view(b, n * inputs_pg.shape[1], inputs_pg.shape[2])
        

        inputs_pg_reshaped = inputs_pg.view(b, n, img_seq_len, img_dim)

        answers = samples["answers"]
        questions = samples["questions"]
        history_actions = samples["history_actions"]

        text_list = []
        trimmed_num_images: List[int] = []
        trimmed_img_starts: List[int] = []
        
        if self.tokenizer.eos_token is None:
            eos_token_str = self.tokenizer.decode(self.tokenizer.eos_token_id)
        else:
            eos_token_str = self.tokenizer.eos_token
        
        for i in range(b):
            cur_num_image = num_images[i]
            if torch.is_tensor(cur_num_image):
                cur_num_image = int(cur_num_image.item())
            else:
                cur_num_image = int(cur_num_image)
            cur_question = questions[i]
            cur_answer = answers[i]
            cur_hist_actions = history_actions[i]

            cur_hist_actions, keep_num_images, img_start_idx = self._trim_history_actions(
                question=cur_question,
                answer=cur_answer,
                history_actions=cur_hist_actions,
                num_images=cur_num_image,
                eos_token_str=eos_token_str,
            )
            trimmed_num_images.append(keep_num_images)
            trimmed_img_starts.append(img_start_idx)

            sep_total = "<|frame_sep|>".join(
                [f"{self.prompt_action}{action}" for action in cur_hist_actions]
            )

            image_start = "<|image_start|>"
            image_end = "<|image_end|>"
            
            cur_text = f"{self.prompt0}\n{cur_question}\n{self.prompt2}\n{image_start}{sep_total}{image_end}\n{self.prompt1}\n{image_start}{image_end}\n{self.prompt_answer}\n{cur_answer}{eos_token_str}"
            text_list.append(cur_text)

        pg_tokens = self.tokenizer(
                text_list,
                return_tensors="pt",
                padding="longest", 
                truncation=True,
                max_length=self.max_txt_len,           
            ).to(image.device)

        inputs_embeds = self.llm.model.embed_tokens(pg_tokens.input_ids)

        new_inputs_embeds = []
        new_attention_masks = []
        new_labels = []

        start_id = self.tokenizer.convert_tokens_to_ids("<|image_start|>")
        end_id = self.tokenizer.convert_tokens_to_ids("<|image_end|>")
        sep_id = self.tokenizer.convert_tokens_to_ids("<|frame_sep|>")


        for i in range(b): 
            cur_txt_embed = inputs_embeds[i] 
            cur_input_ids = pg_tokens.input_ids[i]
            valid_len = int(pg_tokens.attention_mask[i].sum().item())
            cur_input_ids = cur_input_ids[:valid_len]
            cur_txt_embed = cur_txt_embed[:valid_len]

            cur_num_image = int(trimmed_num_images[i])
            img_start_idx = int(trimmed_img_starts[i])
            cur_img_feats = inputs_pg_reshaped[i, img_start_idx:img_start_idx + cur_num_image] # shape: [cur_n_imgs, seq_len, dim]

            cur_answer = answers[i]
            cur_ans_ids = self.tokenizer.encode(cur_answer, add_special_tokens=False)
            cur_ans_ids.append(self.tokenizer.eos_token_id)

            ans_len = len(cur_ans_ids)

            start_indices = (cur_input_ids == start_id).nonzero(as_tuple=True)[0]
            end_indices = (cur_input_ids == end_id).nonzero(as_tuple=True)[0]
            sep_indices = (cur_input_ids == sep_id).nonzero(as_tuple=True)[0]

            hist_start_idx = start_indices[0]
            hist_end_idx = end_indices[0]

            segments = []
            segments.append(cur_txt_embed[:hist_start_idx+1])

            last_idx = hist_start_idx
            img_idx = 0
            for sep_idx in sep_indices:
                segments.append(cur_img_feats[img_idx])
                img_idx += 1
                segments.append(cur_txt_embed[last_idx+1 : sep_idx+1])
                last_idx = sep_idx
            segments.append(cur_img_feats[img_idx])
            img_idx += 1
            segments.append(cur_txt_embed[last_idx+1 : hist_end_idx+1])
            
            segments.append(cur_txt_embed[hist_end_idx+1 : start_indices[1]+1])
            segments.append(cur_img_feats[img_idx])
            segments.append(cur_txt_embed[end_indices[1]:])

            combined_embed = torch.cat(segments, dim=0)
            cur_attention_mask = torch.ones(combined_embed.shape[0], dtype=torch.long, device=image.device)
            cur_label = torch.full((combined_embed.shape[0],), IGNORE_INDEX, dtype=torch.long, device=image.device)
            if ans_len > 0:
                cur_label[-ans_len:] = torch.tensor(cur_ans_ids, dtype=torch.long, device=image.device)
            
            new_inputs_embeds.append(combined_embed)
            new_attention_masks.append(cur_attention_mask)
            new_labels.append(cur_label)

        inputs_embeds = pad_sequence(new_inputs_embeds, batch_first=True, padding_value=0.0)
        attention_mask = pad_sequence(new_attention_masks, batch_first=True, padding_value=0)
        labels = pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX)
        # eos_token_id 45892. 
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )
        loss = outputs.loss

        return {"loss": loss}
