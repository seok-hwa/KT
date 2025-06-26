"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CriterionKT']

class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, featmap):
        n, c, h, w = featmap.shape
        return featmap.reshape((n, c, -1)).softmax(dim=-1)

class ChannelHuberLoss(nn.Module):
    def __init__(self, delta):
        super(ChannelHuberLoss, self).__init__()
        self.delta = delta
    def forward(self, st_featmap, tc_featmap):
        feat_diff = abs(tc_featmap - st_featmap).mean()
        if feat_diff < self.delta:
            loss = self.delta*((tc_featmap-st_featmap)**2)
        else:
            loss = self.delta*(abs(tc_featmap-st_featmap)-0.5*self.delta)
        return loss.sum()

class CriterionKT(nn.Module):
    def __init__(self, s_channels, t_channels, norm_type='none', divergence='mse', temperature=1.0, delta=1.0):
        super(CriterionKT, self).__init__()
        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type == 'spatial':
            self.normalize = nn.Softmax(dim=1)
        self.norm_type = norm_type

        # define loss function
        if divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
        elif divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'l1':
            self.criterion = nn.L1Loss(reduction='sum')
        elif divergence == 'huber':
            # self.criterion = nn.HuberLoss(reduction='sum', delta=self.delta)
            self.criterion = ChannelHuberLoss(delta)

        self.temperature = temperature
        self.divergence = divergence
        self.conv = nn.Conv2d(s_channels, t_channels, kernel_size=1, bias=False)

    def forward(self, preds_S, preds_T):
        n, c, h, w = preds_S.shape

        if preds_S.size(1) != preds_T.size(1):
            preds_S = self.conv(preds_S)

        if self.divergence == 'kl':
            norm_s = self.normalize(preds_S / self.temperature).log()
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        elif self.divergence == 'mse':
            norm_s = F.normalize(preds_S.reshape(n,c,-1), dim=1)
            norm_t = F.normalize(preds_T.reshape(n,c,-1).detach(), dim=1)
        elif self.divergence == 'l1':
            norm_s = F.normalize(preds_S.reshape(n,c,-1), dim=-1)
            norm_t = F.normalize(preds_T.reshape(n,c,-1).detach(), dim=-1)
        elif self.divergence == 'huber':
            norm_s = F.normalize(preds_S.reshape(n, c, -1), dim=1)
            norm_t = F.normalize(preds_T.reshape(n, c, -1).detach(), dim=1)

        loss = self.criterion(norm_s, norm_t)

        if self.norm_type == 'channel':
            if self.divergence == 'mse':
                loss /= n * c
            elif self.divergence == 'kl':
                loss /= n * c
            elif self.divergence == 'l1':
                loss /= n * c
            elif self.divergence == 'huber':
                loss /= n * c
        else:
            loss /= n * h * w

        if self.divergence == 'kl':
            loss = loss*(self.temperature ** 2)

        return loss