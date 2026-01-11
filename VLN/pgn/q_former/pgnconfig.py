from dataclasses import dataclass
import torch

@dataclass
class PGNMultiModalConfig:
    """
    Config for the PGN multimodal model.

    llm_path:
        Directory containing the OpenPangu weights (config.json, safetensors, tokenizer.model).
    projector_type:
        Vision projector type used inside the BLIP2 Q-Former front-end, e.g. "linear", "mlp2x_gelu".
    num_query_token:
        Number of BLIP2 Q-Former query tokens per frame.
    image_size:
        Input resolution of the EVA ViT encoder.
    vit_precision:
        Precision flag passed to BLIP2 EVA ViT ("fp16", "fp32", etc.).
    freeze_vision:
        Whether to freeze the BLIP2 visual encoder weights.
    load_llm_pretrained:
        Whether to load LLM weights from llm_path during model construction.
    load_vision_pretrained:
        Whether to load EVA ViT weights during model construction.
    load_qformer_pretrained:
        Whether to load Q-Former weights during model construction.
    max_txt_len:
        Reserved for potential Q-Former text usage (unused for now).
    compress_type:
        How strongly to compress history frames in navigation mode.
        Follows: "grid:2", "grid:4", or "mean".
    """

    llm_path: str = "/data1/pangu_model"
    projector_type: str = "mlp2x_gelu"
    num_query_token: int = 32
    image_size: int = 224
    vit_precision: str = "fp32"  
    freeze_vision: bool = True 
    load_llm_pretrained: bool = True
    load_vision_pretrained: bool = True
    load_qformer_pretrained: bool = True
    max_txt_len: int = 4096  # 设置的长避免截断
    dtype: torch.dtype = torch.float32
    # compress_type: str = "grid:2" # "grid:4" or "mean"


@dataclass
class VisionProjectorConfig:
    """
    Minimal config wrapper for build_vision_projector.

    mm_hidden_size: dimensionality of Q-Former hidden states.
    hidden_size:   target LLM embedding size (e.g. Pangu hidden_size).
    mm_projector_type: projector type, e.g. 'linear', 'mlp2x_gelu'.
    """

    mm_hidden_size: int = 768 
    hidden_size: int = 4096
    mm_projector_type: str = "mlp2x_gelu"
