
import math
import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..diffusion import create_diffusion
from .multimodal_projector import build_vision_projector, build_gen_vision_projector


class TransfusionMetaModel:
    def __init__(self, config):
        super(TransfusionMetaModel, self).__init__(config)

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)

            self.projector = build_vision_projector(config)
            self.gen_projector = build_gen_vision_projector(config)

            self.diffusion = create_diffusion(timestep_respacing="")

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower

        self.config.vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_proj = True
        self.config.projector_type = getattr(model_args, 'projector_type', 'linear')
        self.config.latent_channels = vision_tower.latent_channels
        self.config.patch_size = getattr(model_args, 'patch_size', 2)

        if getattr(self, "projector", None) is None:
            self.projector = build_vision_projector(self.config)
        else:
            for p in self.projector.parameters():
                p.requires_grad = True

        if getattr(self, "gen_projector", None) is None:
            self.gen_projector = build_gen_vision_projector(self.config)
        else:
            for p in self.gen_projector.parameters():
                p.requires_grad = True


class TransfusionMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def encode_images(self, images):
        image_latents = self.get_model().get_vision_tower()(images)
        return image_latents

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

            


            
