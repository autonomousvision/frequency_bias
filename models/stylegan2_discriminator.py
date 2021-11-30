import torch
from .stylegan3 import SG2Discriminator as DBase


class SG2Discriminator(DBase):
    def __init__(self, resolution, label_size=0, divide_channels_by=1, **kwargs):
        default_kwargs = {
            'channel_base': 32768 // divide_channels_by,
            'channel_max': 512 // divide_channels_by,
            'c_dim': label_size+1,
            'img_channels': 3,
        }
        super(SG2Discriminator, self).__init__(img_resolution=resolution, **default_kwargs, **kwargs)

    def forward(self, img, c=None):
        if c is None:
            c = torch.empty(img.shape[0], 0).to(img.device)
        else:
            c = c.unsqueeze(1)
        return super(SG2Discriminator, self).forward(img, c=c, update_emas=False)