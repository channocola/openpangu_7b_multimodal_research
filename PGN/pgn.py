import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

try:
    import torch_npu  
except ModuleNotFoundError:
    # Allow usage on GPU-only environments; NPU is optional.
    torch_npu = None  # type: ignore

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin

from llm.configuration_openpangu_dense import PanguEmbeddedConfig
from llm.modeling_openpangu_dense import PanguEmbeddedForCausalLM
from llm.tokenization_openpangu import PanguTokenizer
from WorkSpace.PGN.multimodel.multimodel import Blip2QformerForNevigate
from pgnconfig import PGNMultiModalConfig

from constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, VIDEO_START_SPECIAL_TOKEN, VIDEO_END_SPECIAL_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN, NAVIGATION_SPECIAL_TOKEN, NAVIGATION_IDENTIFIER, IAMGE_SEPARATOR
)

class PGNMultiModalVLN(nn.Module, GenerationMixin):
    """
    PGN multi-modal model for VLN.
    Components:
      - BLIP2 front-end (EVA ViT + Q-Former) from `multimodel.mutilmodel`,
      - OpenPangu causal LM from `llm/`,

    Usage (single frame per sample):
        model = PGNMultiModalVLN()
        batch = {
            "input_ids": ...,      # [B, T], use IMAGE_TOKEN_INDEX (-200) as image placeholder
            "attention_mask": ..., # [B, T]
            "images": ...,         # [B, 3, H, W] or [B, T_img, 3, H, W]
            "labels": ...,         # [B, T]
        }
        outputs = model(**batch)
    """

    def __init__(
        self,
        config: Optional[PGNMultiModalConfig] = None,
        device: Optional[torch.device] = None, # npu
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = PGNMultiModalConfig()
        self.mm_config = config # 整个多模态模型的配置，由我们自主定义

        # ----- Load OpenPangu LLM -----
        llm_kwargs: Dict[str, Any] = {}
        if dtype is None:
            dtype = self.mm_config.dtype
            llm_kwargs["torch_dtype"] = dtype  
            self.dtype = dtype
        else:
            llm_kwargs["torch_dtype"] = dtype  
            self.dtype = dtype

        self.llm: PanguEmbeddedForCausalLM = PanguEmbeddedForCausalLM.from_pretrained(
            config.llm_path, **llm_kwargs
        )
        if device is not None:
            self.llm.to(device)

        self.llm_config: PanguEmbeddedConfig = self.llm.config # 这是llm的专属配置，从盘古config直接取出，不是我们定义的
        self.hidden_size: int = int(self.llm_config.hidden_size)

        # ----- BLIP2 front-end (vision tower + Q-Former + projector) -----
        self.vision_frontend = Blip2QformerForNevigate( 
            img_size=config.image_size,
            drop_path_rate=0.0,
            use_grad_checkpoint=False,
            vit_precision=config.vit_precision,
            freeze_vit=config.freeze_vision,
            num_query_token=config.num_query_token,
            embed_dim=256, # nothing
            max_txt_len=config.max_txt_len,
            llm_hidden_size=self.hidden_size, # 对齐到盘古
            mm_projector_type=config.projector_type,
        )
        if device is not None:
            self.vision_frontend.to(device)

        # ----- Pangu tokenizer -----
        vocab_file = os.path.join(config.llm_path, "tokenizer.model")
        if not os.path.isfile(vocab_file):
            raise FileNotFoundError(
                f"Could not find tokenizer.model at '{vocab_file}'. "
                "Make sure `llm_path` points to the OpenPangu checkpoint directory."
            )
        
        self.tokenizer = PanguTokenizer(vocab_file=vocab_file) # 加载盘古tokenizer 

    def update_prompt(self, prompts=None):
        """
        Cache prompts for navigation detection.
        If `prepare_inputs_labels_for_multimodal` is called without an explicit
        `prompts` argument, it will fall back to `self.prompts` (if set).
        """
        self.prompts = prompts
        
    # Attention!
    # Because the Q-Former makes it impossible to take the square root of Query_token_num=32, 
    # and the model mandates that the length of the input Prompt list must be 1, this model does not perform any compression of the visual (query) tokens. 
    # Additionally, the Pangu Big Language Model currently supports a maximum context length of approximately 32768 tokens.
    @property
    def config(self):
        return self.llm_config

    @property
    def device(self) -> torch.device:
        # Use LLM device as the canonical device for this wrapper.
        return next(self.llm.parameters()).device
    
    def cut_nav_token(
        self,
        vis_embed,
        image_counts = None,
        navigation = False,
    ):
        if image_counts is None or (image_counts == 1 and not navigation):
            return vis_embed
        elif navigation:
            vis_embed_nav = vis_embed[-1:]
            return vis_embed, vis_embed_nav
        else:
            return vis_embed, None
        
            
    def visual_token_generation( 
        self,
        images: torch.Tensor,
        prompts: Optional[List[List[str]]] = None,
        image_counts: Optional[List[int]] = None,
        long_video: bool = False,
    ) -> Tuple[List[torch.Tensor], List[bool], List[Optional[torch.Tensor]]]:
        
        # 实际上是query_token_generation ， 
        # 另外请注意方法中部分硬编码为32的地方， 是根据blip_q-former一帧输出的query_tokens数为32来的
        """Encode images + navigation metadata without spatial compression."""
        
        if long_video:
            # use pre-computed features
            image_features = images
        else:
            outputs = self.vision_frontend(images, project_to_llm=True)  # 注意，此处的image_features 实际上是 query_features
            image_features = outputs["llm_inputs"]  # [B*T, Q, H]

        if image_counts is None:
            assert len(image_features) == len(prompts), f"Size mismatch! image_features: {len(image_features)}, prompts: {len(prompts)}"
            # len(prompts) 代表了 batch_size
        else:
            assert len(prompts) == len(image_counts), f"Size mismatch! prompts: {len(prompts)}, image_counts: {len(image_counts)}"

        img_feat_lst = []
        video_or_not = [] # .append :bool
        nav_or_not = [] # .append :nav_token
        final_token_length_lst = []
        total_count = 0
        
        for _idx, prompt in enumerate(prompts):
            assert isinstance(prompt, list), f"Prompt should be a list, but got {type(prompt)}"  # prompt 应该也是一个list，而且与batch_size 即len(image_counts) 所表征的样本数对齐

            if image_counts is None:
                img_feat_prompt = image_features[_idx, None]
            else:
                # 切片，找出对应prompt的图像/query特征
                img_feat_prompt = image_features[total_count:total_count + image_counts[_idx]]
                total_count += image_counts[_idx]
            
            is_navigation = NAVIGATION_IDENTIFIER in prompt[0]
            if is_navigation:
                if image_counts is None or image_counts[_idx] < 1 or len(prompt) != 1:   
                    # 关键在len(prompt) != 1 ， prompts是个list， 其中的每个prompt必须只有一个， 通俗的讲就是只能有一句话
                    raise ValueError('[Navigation] wrong')

            # 本模型不用压缩，保持所有帧的tokens数都是32，而且32也无法被开方，做不了2d池化
            
            final_token, final_token_nav = self.cut_nav_token(
                img_feat_prompt,
                image_counts=None if image_counts is None else image_counts[_idx],
                navigation=is_navigation
            )

            if is_navigation and final_token_nav is None:
                raise ValueError('[Navigation] wrong')
            
            final_token = final_token[None].expand(len(prompt), -1, -1, -1).flatten(1, 2)  # [1, T*query_token_num, hidden] --> llm不关心几张图片
                                                                                           # 注意此处是len(prompt) 并非len(prompts)
            if image_counts is not None:
                if is_navigation: 
                    final_token_nav = final_token_nav[None].expand(len(prompt), -1, -1, -1).flatten(1, 2)
                    assert final_token_nav.shape[0] == 1 and final_token_nav.shape[1] == 32 and final_token.shape[0] == 1
                    # 注意： 所有的token 无论final_token_nav还是final_token .shape[1] == 32
                    nav_or_not.append(final_token_nav)
                else:
                    nav_or_not.append(None) # 无images_count 不导航

                if image_counts[_idx] == 1: # 恰好只有一帧
                    if is_navigation: # 恰好是导航模式下的第一帧
                        video_or_not.append(True)  # 导航会被看作是一种视频
                    else:
                        video_or_not.append(False)
                else:
                    video_or_not.append(True)
            else:
                video_or_not.append(False) # 无images_count 不导航
                nav_or_not.append(None)
            img_feat_lst.append(final_token)

        return img_feat_lst, video_or_not, nav_or_not       # img_feat_lst 是batch_size个样本堆叠生成的链表，其中一个样本可能包含多帧，但F*Q展平
                                                            # video_or_not 是batch_size个bool
                                                            # nav_or_not 是一个样本在导航模式下的最后一帧


    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images, prompts=None, long_video=False):
        """
        Expected input:
        images: either a multi-sample, multi-frame input of [batch_size, num_frames, c, h, w], 
                or a multi-sample, single-frame input of [batch_size, c, h, w], 
                or the preprocessed visual feature sequence when long_video=True.
        input_ids: 
        """
        if prompts is None and hasattr(self, 'prompts'):
            prompts = self.prompts

        embed_tokens = self.llm.get_input_embeddings()  # Pangu embeddings
        vision_frontend = self.vision_frontend  # Blip_Q-Former

        if vision_frontend is None or images is None or input_ids.shape[1] == 1:  # cache casullm在生成每一个token时不必再重新过一遍vision_frontend
            if past_key_values is not None and vision_frontend is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        # pre-process images for long video
        # 注意此处，以宽高是否大于1000判断是否为长视频
        if images[0].shape[-1] > 1000:
            long_video = True
        else:
            long_video = False

        if type(images) is list or images.ndim == 5:
            """
            # 处理多图像 / 视频输入：
            # 若 images 是 list,或是一个 5D Tensor(形状 [B, T, C, H, W])，则进入该分支。
            # 典型情况：
            #   - 图像列表： [ [C,H,W], [T,C,H,W], ... ]
            #   - 视频批次： images.ndim == 5 → [B, T, C, H, W]
            # 处理逻辑说明：
            #   如果 images 是 5D Tensor (如 [B,T,C,H,W]):
            #       Python 会迭代第 0 维，将其拆成 B 个 [T,C,H,W] Tensor
            #       即变成一个 list: [ [T,C,H,W], [T,C,H,W], ... ]
            #   接着统一保证每个元素都是 4D([F,C,H,W])
            #       若单张图像 [C,H,W] → unsqueeze → [1,C,H,W]
            #   image_counts 用于记录每个样本的帧数 F,用于后续在 encode_images 中还原结构
            #       例如输入 [2,4,3,224,224] → image_counts = [4,4]
            #   torch.cat(images, dim=0) 会将所有帧沿第 0 维拼接：
            #       输出 concat_images 形状为 [sum(Fi), C, H, W]
            #       即所有样本的视频帧展平后喂给视觉编码器，提高处理效率
            # 最终结构示例(images.ndim == 5 且 images=[2,4,3,224,224]):
            #   images        → [[4,3,224,224], [4,3,224,224]]
            #   image_counts  → [4, 4]
            #   concat_images → [8,3,224,224]
            # not reseshape for long video
            """
            if not long_video:
                images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images]
            image_counts = [image.shape[0] for image in images]
            concat_images = torch.cat(images, dim=0)
            image_features, video_or_not, nav_or_not = self.visual_token_generation(concat_images, prompts, image_counts, long_video=long_video)
            # type: list list list
        else:
            image_features, video_or_not, nav_or_not = self.visual_token_generation(images, prompts, long_video=long_video)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids): # (Batch_Size, Sequence_Length)
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0: 
                # no images: 拼一个0张量上来冒充visual_embed --> plaese infer deepspeed zero1/2/3
                # attention it please:
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][0]
                else:
                    cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0] # 找图像的占位符 导航模式应该只有一个图像占位符
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            if not long_video:
                token_idx = 0   
                while image_token_indices.numel() > 0:
                    if isinstance(image_features, list):
                        cur_image_features = image_features[cur_image_idx][token_idx]
                        # image_features --> list {<torch.tensor> [1, query_token_nums, hidden], ..... , [1, query_token_nums, hidden] }
                        # token_idx 可能用于非导航的多模态训练时用到索引，导航模式下不起作用 需要在上面visual_token_generation中，len(prompt) > 1
                        # cur_image_idx 就是样本索引 batch_size索引
                    else:
                        cur_image_features = image_features[cur_image_idx]
                    image_token_start = image_token_indices[0] # <image> 起始

                    if getattr(self.mm_config, 'tune_mm_mlp_adapter', False) and getattr(self.mm_config, 'mm_use_im_start_end', False):
                    # 不使用 MLP adapter（大多数 BLIP2 / EVA ViT + Q-Former 配置都是）
                    # 不使用 <im_start> <im_end> token 包围方式，或者只使用一种
                        raise ValueError('wrong')
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[image_token_start + 1:image_token_start + 2]))
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(
                                torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                           dtype=labels.dtype))
                            cur_new_labels.append(cur_labels[image_token_start:image_token_start + 1])
                            cur_labels = cur_labels[image_token_start + 2:]
                    else: 
                        if nav_or_not[cur_image_idx] is None and video_or_not[cur_image_idx] is False:
                            # 没有导航帧 ，也不是视频（导航应该被看做是一种视频）
                            # id : text_token_id + <iamge_id> (.....)
                            # text_token_embeds + single_image_token_embeds (.....)
                            cur_new_input_embeds.append(embed_tokens(cur_input_ids[:image_token_start]))
                            cur_new_input_embeds.append(cur_image_features)
                            assert cur_image_features.shape[0] == 32
                        elif nav_or_not[cur_image_idx] is None and video_or_not[cur_image_idx] is True:
                            # 不是导航帧， 但是是视频
                            # id : text_token_id + <seperator_token_id> + <iamge_token_id> (.....)
                            # text_token_embeds + image_token_embeds + seperator_token_embeds + image_token_embeds + ......   (.....)
                            cur_new_input_embeds.append(embed_tokens(cur_input_ids[:image_token_start]))
                            seperator_token = embed_tokens(cur_input_ids[image_token_start - 1, None])
                            video_index = 0
                            assert len(cur_image_features) % 32 == 0 # hard code  --> 32 
                            
                            for ii in range(int(len(cur_image_features) / 32)):
                                cur_new_input_embeds.append(cur_image_features[video_index:video_index + 32])
                                if ii == (len(cur_image_features) / 32) - 1:
                                    break
                                cur_new_input_embeds.append(seperator_token)
                                video_index += 32

                        else:
                            # 导航
                            # id : text_token_id + <seperator_token_id> + <iamge_token_id> + <iamge_token_id+1 : ?> + <iamge_token_id+2 : ?> + <iamge_token_id+1 : end>
                            # text_token_embeds + image_token_embeds + seperator_token_embeds + image_token_embeds + seperator_token_embeds + image_token_embeds + .... + iamge_token_+1 :? + iamge_token_+2 :? + nav_token
                            assert video_or_not[cur_image_idx] is True   # 导航必须是视频
                            assert token_idx == 0   # 确保 在导航模式下token_idx == 0 也就是只有一个<image_token_id>  len(prompt) == 1 只是一句话，在文本+视频之后将不会有任何东西 
                            assert nav_or_not[cur_image_idx][token_idx].shape[0] == 32 
                            cur_new_input_embeds.append(embed_tokens(cur_input_ids[:image_token_start]))
                            seperator_token = embed_tokens(cur_input_ids[image_token_start - 1, None])
                            video_index = 0
                            assert len(cur_image_features) % 32 == 0
                            for ii in range(int(len(cur_image_features) / 32)):
                                cur_new_input_embeds.append(cur_image_features[video_index:video_index + 32])
                                if ii == (len(cur_image_features) / 32) - 1:
                                    break
                                cur_new_input_embeds.append(seperator_token)
                                video_index += 32
                            cur_new_input_embeds.append(embed_tokens(cur_input_ids[image_token_start + 1:image_token_start + 3]))
                            cur_new_input_embeds.append(nav_or_not[cur_image_idx][token_idx])

                        if labels is not None:
                            # labels 应该需要单独传入？
                            if nav_or_not[cur_image_idx] is None and video_or_not[cur_image_idx] is False:
                                cur_new_labels.append(cur_labels[:image_token_start])
                                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                               dtype=labels.dtype))
                                cur_labels = cur_labels[image_token_start + 1:]  
                            elif nav_or_not[cur_image_idx] is None and video_or_not[cur_image_idx] is True:
                                cur_new_labels.append(cur_labels[:image_token_start])
                                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                               dtype=labels.dtype))
                                cur_new_labels.append(torch.full((int(cur_image_features.shape[0] / 32 - 1),), IGNORE_INDEX, 
                                               device=labels.device, dtype=labels.dtype))
                                # 有多少张图片就有多少-1个分隔符
                                cur_labels = cur_labels[image_token_start + 1:]
                            else:
                                cur_new_labels.append(cur_labels[:image_token_start])
                                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                               dtype=labels.dtype))
                                cur_new_labels.append(torch.full((int(cur_image_features.shape[0] / 32 - 1),), IGNORE_INDEX,
                                               device=labels.device, dtype=labels.dtype))
                                # 有多少张图片就有多少-1个分隔符
                                cur_new_labels.append(torch.full((nav_or_not[cur_image_idx][token_idx].shape[0] + 2,), IGNORE_INDEX,
                                               device=labels.device, dtype=labels.dtype))
                                # nav 多3个ignore_label
                                cur_labels = cur_labels[image_token_start + 3:]

                    if getattr(self.mm_config, 'tune_mm_mlp_adapter', False) and getattr(self.mm_config, 'mm_use_im_start_end', False):
                        raise ValueError('wrong')

                    else:
                        if nav_or_not[cur_image_idx] is not None:
                            cur_input_ids = cur_input_ids[image_token_start + 3:]  # 导航就不应该切片
                        else:
                            cur_input_ids = cur_input_ids[image_token_start + 1:]  # 切片，切之后可能会有的input_id
                    image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0] # 检索当前<image_id>之后还有没有别的<image_id>
                    token_idx += 1

                cur_image_idx += 1  # 准备处理下一个样本
                if cur_input_ids.numel() > 0:  # 
                    if getattr(self.mm_config, 'tune_mm_mlp_adapter', False) and getattr(self.mm_config, 'mm_use_im_start_end', False):
                        # 这个  raise ValueError('wrong') 是我单独加上来的
                        raise ValueError('wrong')
                        cur_new_input_embeds.append(embed_tokens(cur_input_ids).detach())
                    else:
                        cur_new_input_embeds.append(embed_tokens(cur_input_ids))
                    if labels is not None:
                        cur_new_labels.append(cur_labels)
                
                cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
                cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    cur_new_labels = torch.cat(cur_new_labels, dim=0)
                    assert cur_new_input_embeds.shape[0] == cur_new_labels.shape[0]
                    new_labels.append(cur_new_labels)

            else: # 就是长视频（注意是否长视频和是否视频的概念是不一样的） 直接输入图像特征
                cur_new_input_embeds = torch.Tensor(len(cur_input_ids), self.config.hidden_size).to(dtype=self.dtype, device=self.device)
                text_token_indices = torch.where(cur_input_ids != IMAGE_TOKEN_INDEX)[0]
                if not self.training and embed_tokens.weight.device != cur_input_ids.device:
                    model_device = embed_tokens.weight.device
                    data_device = cur_input_ids.device
                    cur_input_ids_text = cur_input_ids[text_token_indices].to(device=model_device)
                    cur_new_input_embeds[text_token_indices] = embed_tokens(cur_input_ids_text).to(device=data_device)
                else:
                    cur_new_input_embeds[text_token_indices] = embed_tokens(cur_input_ids[text_token_indices])
                cur_image_features = image_features[cur_image_idx]
                cur_new_input_embeds[image_token_indices] = cur_image_features
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    new_labels.append(cur_labels)
                cur_image_idx += 1

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):  # 对齐形成大张量用的
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)
            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def initialize_tokenizer(self, tokenizer):
        """
        简单迁移,尚不明确以下几个参数的作用：
            tune_mm_mlp_adapter
            pretrain_mm_mlp_adapter
            可能涉及某种微调方式
        需要传入的tokenizer就是盘古的tokenizer
        """
        if tokenizer == None:
            tokenizer = self.tokenizer
        else:
            tokenizer = tokenizer

        tokenizer.add_tokens([VIDEO_START_SPECIAL_TOKEN, VIDEO_END_SPECIAL_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN, NAVIGATION_SPECIAL_TOKEN, IAMGE_SEPARATOR], special_tokens=True)
        self.llm.resize_token_embeddings(len(tokenizer)) # 给token_embeddings扩容
        if self.mm_config.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.llm.resize_token_embeddings(len(tokenizer))
        
        # 注意：
        # 以下代码需要在pgnconfig中（即self.mm_config）进行配置才会进入分支，否则默认全为False，下面代码将全部跳过：
        if self.mm_config.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True) # 以patch_token为显式传入，没什么用
            self.llm.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.llm.get_input_embeddings().weight.data
                output_embeddings = self.llm.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if self.mm_config.tune_mm_mlp_adapter: # 大语言模型全冻结，只训练 mm_projector（以及某些场景下的 embedding） 时会用到
                for p in self.llm.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.llm.get_input_embeddings().parameters():
                    p.requires_grad = False  # 前后矛盾，可能是历史遗留错误

            if self.mm_config.pretrain_mm_mlp_adapter is not None: # 即使有也用不上，维度对不齐
                mm_projector_weights = torch.load(self.mm_config.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif self.mm_config.mm_use_im_patch_token:
            if self.mm_config.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs
    
    def forward(self, 
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                images: Optional[torch.FloatTensor] = None,
                prompts: Optional[List[str]] = None,
                return_dict: Optional[bool] = None,
            ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not self.training:
            if images[0].device != self.device:
                images[0] = images[0].to(device=self.device)
            if input_ids.device != self.device:
                input_ids = input_ids.to(device=self.device)

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, prompts=prompts)

        if torch_npu is not None:
            torch.npu.empty_cache()  
        else:
            torch.cuda.empty_cache()    # 默认npu不可用使用cuda

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.llm.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


