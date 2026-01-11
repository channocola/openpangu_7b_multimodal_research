import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch_npu  

import torch.nn as nn
from torch.nn import functional as F

from .blip2 import (
    Blip2Base,
    disabled_train,
)

from .projector import build_vision_projector
from .pgnconfig import VisionProjectorConfig


# 定义如下两个类，来源blip2-qformer 两个类区别仅限于有无投影层

class Blip2QformerForPanGuMM(Blip2Base):
    """
    BLIP2 front-end (ViT + Q-Former) without any language model.

    This class is meant to be used as the visual front-end for an
    external LLM (e.g. openPangu). It handles:
      - tokenizer initialization (BERT tokenizer used by Q-Former),
      - EVA ViT visual encoder,
      - Q-Former with cross-attention to visual features,
    """
    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=True,
        num_query_token=32,
        load_vision_pretrained=True,
        load_qformer_pretrained=True,
    ):
        super().__init__()
        # Visual encoder (EVA ViT from LAVIS)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            img_size,
            drop_path_rate,
            use_grad_checkpoint,
            vit_precision,
            load_pretrained=load_vision_pretrained,
        )
        if freeze_vit:
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            self.visual_encoder.requires_grad_(False)
            self.ln_vision.requires_grad_(False)
            logging.info("freeze vision encoder")

        # Q-Former with cross-attention to the visual encoder
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token,
            self.visual_encoder.num_features,
            load_pretrained=load_qformer_pretrained,
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
            
    def encode_image(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode images into Q-Former query features.

        Args:
            image: Tensor of shape [B, 3, H, W].

        Returns:
            q_hidden:     Q-Former output hidden states for query tokens,
                          [B, num_query_token, C_q].
        """
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        q_hidden = query_output.last_hidden_state
        return q_hidden

    def forward(
        self, image: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Forward pass for convenience.

        Args:
            image: Tensor of shape [B, 3, H, W].

        Returns:
              - "q_hidden": Q-Former query outputs, [B, num_query_token, C_q].
        """
        q_hidden = self.encode_image(image)

        return q_hidden



class Blip2QformerForNevigate(Blip2Base):
    # 自带投影层的nevigate式qformer，加载权重时需要注意
    # TODO 尚未使用
    """
    BLIP2 front-end (ViT + Q-Former) without any language model.

    This class is meant to be used as the visual front-end for an
    external LLM (e.g. openPangu). It handles:
      - tokenizer initialization (BERT tokenizer used by Q-Former),
      - EVA ViT visual encoder,
      - Q-Former with cross-attention to visual features,
      - an optional projector that maps Q-Former outputs to an LLM
        hidden size via build_vision_projector.
    """

    def __init__(
        self,
        img_size: int = 224,
        drop_path_rate: float = 0.0,
        use_grad_checkpoint: bool = False,
        vit_precision: str = "fp16",
        freeze_vit: bool = True,
        num_query_token: int = 32,
        llm_hidden_size: Optional[int] = 4096,
        load_vision_pretrained: bool = True,
        load_qformer_pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Visual encoder (EVA ViT from LAVIS)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            img_size,
            drop_path_rate,
            use_grad_checkpoint,
            vit_precision,
            load_pretrained=load_vision_pretrained,
        )
        if freeze_vit:
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            self.visual_encoder.requires_grad_(False)
            self.ln_vision.requires_grad_(False)
            logging.info("freeze vision encoder")

        # Q-Former with cross-attention to the visual encoder
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token,
            self.visual_encoder.num_features,
            load_pretrained=load_qformer_pretrained,
        )
        
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # Optional projector to match an external LLM hidden size (e.g. Pangu)
        self.mm_projector: Optional[nn.Module]
        if llm_hidden_size is not None:
            proj_cfg = VisionProjectorConfig()
            self.mm_projector = build_vision_projector(proj_cfg)
        else:
            self.mm_projector = None

    def encode_image(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode images into Q-Former query features.

        Args:
            image: Tensor of shape [B, 3, H, W].

        Returns:
            q_hidden:     Q-Former output hidden states for query tokens,
                          [B, num_query_token, C_q].
        """
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        q_hidden = query_output.last_hidden_state
        return q_hidden

    def forward(
        self, image: torch.Tensor, project_to_llm: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass for convenience.

        Args:
            image: Tensor of shape [B, 3, H, W].
            project_to_llm: if True and mm_projector is available,
                also return features projected to the LLM hidden size.

        Returns:
            A dict with keys:
              - "q_hidden": Q-Former query outputs, [B, num_query_token, C_q].
              - "llm_inputs" (optional): features mapped to LLM hidden size,
                    [B, num_query_token, hidden_size] if project_to_llm is True
                    and mm_projector is not None.
        """
        q_hidden = self.encode_image(image)

        outputs: Dict[str, Any] = {
            "q_hidden": q_hidden,
        }

        if project_to_llm and self.mm_projector is not None:
            outputs["llm_inputs"] = self.mm_projector(q_hidden)

        return outputs

