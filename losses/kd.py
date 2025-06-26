import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['CriterionKD']

class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''
    def __init__(self, temperature=1):
        super(CriterionKD, self).__init__()
        self.temperature = temperature

    def forward(self, pred, soft):
        B, C, h, w = soft.size()

        # scale_pred = pred.permute(0,2,3,1).contiguous().view(-1,C)
        # scale_soft = soft.permute(0,2,3,1).contiguous().view(-1,C)
        # p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
        # p_t = F.softmax(scale_soft / self.temperature, dim=1)
        # loss = F.kl_div(p_s, p_t.detach(), reduction='batchmean') * (self.temperature**2)

        scale_pred = pred.contiguous().view(B,C,-1)
        scale_soft = soft.contiguous().view(B,C,-1)
        p_s = F.log_softmax(scale_pred / self.temperature, dim=-1)
        p_t = F.softmax(scale_soft / self.temperature, dim=-1)
        loss = F.kl_div(p_s, p_t.detach(), reduction='sum') * (self.temperature ** 2) / (B * C)

        return loss