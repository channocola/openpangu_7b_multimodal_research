import re
from typing import Any

import torch
import torch.nn as nn
import torch_npu 

class IdentityMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        return x + self.proj(x)


def _resolve_dims(config: Any, **kwargs) -> tuple[int, int]:
    """
    Resolve (mm_hidden_size, hidden_size) either from a config-like object
    or from explicit kwargs. This keeps compatibility with NaVid configs
    and allows simple dataclass configs (e.g. VisionProjectorConfig).
    """
    mm_hidden_size = getattr(config, "mm_hidden_size", None)
    hidden_size = getattr(config, "hidden_size", None)

    if mm_hidden_size is None:
        mm_hidden_size = kwargs.get("mm_hidden_size")
    if hidden_size is None:
        hidden_size = kwargs.get("hidden_size")

    if mm_hidden_size is None or hidden_size is None:
        raise ValueError(
            "build_vision_projector requires 'mm_hidden_size' and 'hidden_size' "
            "either as attributes of config or as keyword arguments."
        )

    return int(mm_hidden_size), int(hidden_size)


def build_vision_projector(config: Any, delay_load: bool = False, **kwargs) -> nn.Module:
    """
    Build a vision projector module.

    - In the NaVid setting, `config` is typically the LLM config with
      attributes `mm_hidden_size`, `hidden_size`, and optional
      `mm_projector_type`.
    - In the BLIP2+Pangu setting, a lightweight config (e.g. VisionProjectorConfig)
      can be passed, as long as it exposes the same attributes or they are
      provided via kwargs.

    The projector operates on the last dimension of the input tensor, so it
    supports inputs shaped like [B, T, mm_hidden_size], including Q-Former
    outputs `q_hidden` from the BLIP2 front-end.
    """
    projector_type = getattr(config, "mm_projector_type", kwargs.get("mm_projector_type", "linear"))
    mm_hidden_size, hidden_size = _resolve_dims(config, **kwargs)

    if projector_type == "linear":
        return nn.Linear(mm_hidden_size, hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
