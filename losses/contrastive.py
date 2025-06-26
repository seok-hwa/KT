"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CriterionContrastive']

def _factor_transfer(x, y, p=1):
    assert x.dim() == y.dim() == 4
    x = F.normalize(x.view(x.size(0), -1), p=2, dim=1)
    y = F.normalize(y.view(y.size(0), -1), p=2, dim=1)
    diff = x - y
    diff = diff.norm(p=p).pow(p) / diff.numel()
    return diff.pow(1/p)

class CriterionContrastive(nn.Module):
    def __init__(self, p=1, negative=False):
        super(CriterionContrastive, self).__init__()
        self.p = p
        self.negative = negative

    def forward(self, x, y):
        if self.negative:
            return  -_factor_transfer(x, y, p=self.p)
        else:
            return _factor_transfer(x, y, p=self.p)