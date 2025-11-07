import copy
import os
from dataclasses import dataclass, field

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    vision_tower: Optional[str] = field(default="stabilityai/sdxl-vae")
    projector_type: Optional[str] = field(default="linear")
    use_im_start_end: bool = field(default=True)
    patch_size: int = field(default=2)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    image_size: int = field(default=256)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


def train(attn_implementation=None):
    pass


if __name__ == "__main__":
    train()
