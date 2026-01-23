import torch
import torch.nn as nn
import torch.nn.functional as F

import model_loader
from model_loader import *

class MLP(nn.Module):
    def __init__(self, dim: int, config: MLP):
        super().__init__()

        self.layer = nn.ModuleList()

        current_dim: int = dim

        for idx, lay in enumerate(config.sequence):
            if isinstance(lay, LinearStep):
                in_dim, out_dim = dim, dim

                if idx == 0:
                    out_dim *= config.multiplier
                elif idx == len(config.sequence) - 1:
                    in_dim *= config.multiplier
                else:
                    in_dim *= config.multiplier
                    out_dim *= config.multiplier
                self.layer.append(nn.Linear(int(in_dim), int(out_dim), bias=lay.linear.bias))
                current_dim = out_dim

            elif isinstance(lay, NormComponent):
                self.layer.append(get_norm(lay.norm, int(current_dim)))

            elif isinstance(lay, ActivationComponent):
                self.layer.append(get_activation(lay.activation.type))

            else:
                raise ValueError(f"Unknown MLP layer type: {type(lay)}")

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

class LearnableRMSNorm(nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.weight, eps=self.eps)

class StaticRMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()

        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class SquaredReLU(nn.Module):
    def forward(self, x):
        return F.relu(x).square()


def get_norm(norm: Norm, dim: int) -> nn.Module:
    if isinstance(norm, model_loader.RMSNorm):
        if norm.learnable_gamma:
            return LearnableRMSNorm(dim)
        else:
            return StaticRMSNorm()

    if isinstance(norm, LayerNorm):
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown norm config: {type(norm)}")

def get_activation(act: str) -> nn.Module:
    if act == "gelu":
        return nn.GELU(approximate="tanh")
    if act == "relu":
        return nn.ReLU()
    if act == "squared_relu":
        return SquaredReLU()
    if act == "silu":
        return nn.SiLU()
    if act == "tanh":
        return nn.Tanh()
    if act == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation config: {act}")