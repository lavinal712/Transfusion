import os

from .vae_encoder import VAEVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'vision_tower', None)
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("stabilityai") or \
        vision_tower.startswith("black-forest-labs") or "vae" in vision_tower.lower():
        return VAEVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
