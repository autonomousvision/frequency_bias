import torch
from .stylegan3 import SG2Generator as GBase


class SG2Generator(GBase):
    def __init__(self, resolution, z_dim, divide_channels_by=1, **kwargs):
        default_kwargs = {
            'channel_base': 32768 // divide_channels_by,
            'channel_max': 512 // divide_channels_by,
            'c_dim': 0,
            'w_dim': z_dim,
            'img_channels': 3,
        }
        super(SG2Generator, self).__init__(img_resolution=resolution, z_dim=z_dim, **default_kwargs, **kwargs)

    def forward(self, z):
        c = torch.empty(z.shape[0], 0).to(z.device)
        return super(SG2Generator, self).forward(z, c=c, truncation_psi=1, truncation_cutoff=None, update_emas=False)