import torch
from .misc import to_image


def psnr(pred, target):
    pred = to_image(pred)
    target = to_image(target)

    mse = torch.nn.functional.mse_loss(pred.to(torch.float), target.to(torch.float), reduction='none')
    mse = mse.mean([1,2,3])
    max_i2 = 255**2

    return 10 * torch.log10(max_i2/mse)