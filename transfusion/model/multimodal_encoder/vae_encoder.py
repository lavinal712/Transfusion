import torch
import torch.nn as nn

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor


class VAEVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoencoderKL.load_config(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        try:
            self.vision_tower = AutoencoderKL.from_pretrained(self.vision_tower_name, device_map=device_map)
        except:
            self.vision_tower = AutoencoderKL.from_pretrained(self.vision_tower_name, subfolder="vae", device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.scaling_factor = self.vision_tower.config.scaling_factor
        self.vae_scale_factor = 2 ** (len(self.vision_tower.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            latents = []
            for image in images:
                latent = self.vision_tower(image.to(self.device, dtype=self.dtype).unsqueeze(0)).sample
                latents.append(latent)
        else:
            latents = self.vision_tower(images.to(self.device, dtype=self.dtype)).sample

        return latents

    @torch.no_grad()
    def encode(self, images):
        return self.vision_tower.encode(images.to(self.device, dtype=self.dtype)).latent_dist.sample().mul_(self.scaling_factor)

    @torch.no_grad()
    def decode(self, latents):
        return self.vision_tower.decode(latents / self.scaling_factor).sample

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return 1024 # dummy

    @property
    def latent_channels(self):
        return self.config.latent_channels
