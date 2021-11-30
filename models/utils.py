import torch
import torch.nn.functional as F


def up_bilinear(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)


def up_nearest(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode='nearest')


def up_zeros(x, scale_factor):
    if scale_factor != 2:
        raise NotImplementedError
    assert x.ndim == 4
    uph = torch.stack([x, torch.zeros_like(x)], dim=-1).flatten(-2, -1)
    uphw = torch.stack([uph, torch.zeros_like(uph)], dim=-2).flatten(-3, -2)
    return uphw


def up_shuffle(x, scale_factor):
    return F.pixel_shuffle(x, upscale_factor=scale_factor)


UPSAMPLE_FNS = {
    'bilinear': up_bilinear,
    'nearest': up_nearest,
    'zeros': up_zeros,
    'shuffle': up_shuffle,
}


def down_avg(x, scale_factor):
    return F.avg_pool2d(x, kernel_size=scale_factor, stride=scale_factor, padding=0)


def down_stride(x, scale_factor):
    if scale_factor != 2:
        raise NotImplementedError
    assert x.ndim == 4
    return x[..., ::2, ::2]


def down_blurpool(x, scale_factor):
    f = torch.tensor([[1, 3, 1], [3, 9, 3], [1, 3, 1]], dtype=x.dtype, device=x.device) / 25.
    f = f.flip(list(range(f.ndim)))

    # Pad input with reflection padding
    x = torch.nn.functional.pad(x, (1,1,1,1), mode='reflect')

    # Convolve with the filter to filter high frequencies.
    num_channels = x.shape[1]
    f = f.view(1, 1, *f.shape).repeat(num_channels, 1, 1, 1)
    x = F.conv2d(input=x, weight=f, groups=num_channels)

    return down_avg(x, scale_factor)


DOWNSAMPLE_FNS = {
    'avg': down_avg,
    'stride': down_stride,
    'blurpool': down_blurpool,
}