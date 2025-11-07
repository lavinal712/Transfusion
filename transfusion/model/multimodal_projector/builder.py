import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'projector_type', 'linear')

    if projector_type == 'linear':
        return PatchEmbed(config.image_size, config.patch_size, config.in_channels, config.hidden_size)

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_gen_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'projector_type', 'linear')

    if projector_type == 'linear':
        return FinalLayer(config.hidden_size, config.patch_size, config.out_channels)

    raise ValueError(f'Unknown projector type: {projector_type}')
