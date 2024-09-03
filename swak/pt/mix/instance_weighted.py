from typing import Self, Any
import torch as pt
import torch.nn as ptn

from ..types import Tensor, Module


class ArgsInstanceWeightedSumMixer(Module):

    def __init__(self, mod_dim: int, n_features: int, **kwargs: Any) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.n_features = n_features
        self.rotate = ptn.Linear(n_features * mod_dim, n_features, **kwargs)
        self.norm = ptn.Softmax(dim=-1)

    def forward(self, *inps: Tensor) -> Tensor:
        coefficients = self.norm(self.rotate(pt.cat(inps, dim=-1)))
        return (coefficients.unsqueeze(-2) @ pt.stack(inps, -2)).squeeze(-2)


class StackInstanceWeightedSumMixer(Module):

    def __init__(self, mod_dim: int, n_features: int, **kwargs: Any) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.n_features = n_features
        self.rotate = ptn.Linear(n_features * mod_dim, n_features, **kwargs)
        self.norm = ptn.Softmax(dim=0)

    def forward(self, inp: Tensor) -> Tensor:
        coefficients = self.norm(self.rotate(inp.flatten(start_dim=-2)))
        return (coefficients.unsqueeze(dim=-2) @ inp).squeeze(dim=-2)
