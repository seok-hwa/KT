"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CriterionHuber']

class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, featmap):
        n, c, h, w = featmap.shape
        return featmap.reshape((n, c, -1)).softmax(dim=-1)

class CriterionHuber(nn.Module):
    def __init__(self, s_channels, t_channels, norm_type='none',):
        super(CriterionHuber, self).__init__()
        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type == 'spatial':
            self.normalize = nn.Softmax(dim=1)
        self.norm_type = norm_type

        # define loss function
        self.L2_criterion = nn.MSELoss(reduction='sum')
        self.L1_criterion = nn.L1Loss(reduction='sum')

        self.conv = nn.Conv2d(s_channels, t_channels, kernel_size=1, bias=False)

    def forward(self, preds_S, preds_T, similarity):
        n, c, h, w = preds_S.shape

        if preds_S.size(1) != preds_T.size(1):
            preds_S = self.conv(preds_S)

        norm_s = F.normalize(preds_S.reshape(n, c, -1), dim=1)
        norm_t = F.normalize(preds_T.reshape(n, c, -1).detach(), dim=1)

        if similarity <= 0:
            loss = self.L2_criterion(norm_s, norm_t)
        else:
            loss = self.L1_criterion(norm_s, norm_t)

        loss /= n * h * w

        return loss