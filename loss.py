import torch
from utils.misc import construct_class_by_name
from utils.spectrum import get_spectrum
from torch import Tensor


class MSESpectrumLoss(torch.nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super(MSESpectrumLoss, self).__init__(*args, **kwargs)

    @staticmethod
    def get_log_spectrum(input):
        spectra = get_spectrum(input.flatten(0, 1)).unflatten(0, input.shape[:2])
        spectra = spectra.mean(dim=1)             # average over channels
        return (1 + spectra).log()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_spectrum = self.get_log_spectrum(input)
        target_spectrum = self.get_log_spectrum(target)
        return super(MSESpectrumLoss, self).forward(input_spectrum, target_spectrum)


class MultiLoss(object):
    def __init__(self, losses, weights):
        self.losses = losses
        self.weights = weights

    def __call__(self, *args, **kwargs):
        loss = 0
        for loss_fn, w in zip(self.losses, self.weights):
            loss = loss + w * loss_fn(*args, **kwargs)
        return loss


def get_criterion(class_name, weight=None):
    if not isinstance(class_name, list):
        return construct_class_by_name(class_name=class_name)

    if (weight is None) or (len(class_name) != len(weight)):
        raise AttributeError('Number of losses and number of weights have to match.')

    losses = [construct_class_by_name(class_name=n) for n in class_name]
    return MultiLoss(losses, weight)
