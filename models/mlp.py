"""Fully-connected architecture."""

import torch
import torch.nn as nn

__all__ = ['MLP']


class MLP(nn.Module):
    def __init__(self, input_size, output_size, nhidden=3, dhidden=16, activation=nn.ReLU, bias=True):
        super(MLP, self).__init__()
        self.nhidden = nhidden
        if isinstance(dhidden, int):
            dhidden = [dhidden] * (self.nhidden + 1)        # one for input layer

        input_layer = nn.Linear(input_size, dhidden[0], bias=bias)
        hidden_layers = [nn.Linear(dhidden[i], dhidden[i+1], bias=bias) for i in range(nhidden)]
        output_layer = nn.Linear(dhidden[nhidden], output_size, bias=bias)

        layers = [input_layer] + hidden_layers + [output_layer]

        main = []
        for l in layers:
            main.extend([l, activation()])
        main = main[:-1]          # no activation after last layer

        self.main = nn.Sequential(*main)

    def forward(self, x, c=None):
        assert x.ndim == 2
        out = self.main(x)
        if c is not None:
            out = out[range(len(c)), c].unsqueeze(1)
        return out