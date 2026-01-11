import re

from q_former.registry import registry
try:
    from processor.base_processor import BaseProcessor
except ModuleNotFoundError:
    from base_processor import BaseProcessor
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)


class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            return cls()
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)
        image_size = cfg.get("image_size", 224)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)
        return cls(image_size=image_size, mean=mean, std=std)

    def __call__(self, item):
        return self.transform(item)


def build_blip2_image_train_processor(image_size=224, mean=None, std=None, cfg=None):
    if cfg is not None:
        return Blip2ImageTrainProcessor.from_config(cfg)
    return Blip2ImageTrainProcessor(image_size=image_size, mean=mean, std=std)

