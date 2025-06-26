"""Base Model for Semantic Segmentation"""
import torch
import torch.nn as nn
from .base_models.resnet import *
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


__all__ = ['SegBaseModel']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50', local_rank=None, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        self.aux = aux
        self.nclass = nclass
        self.backbone = backbone
        if backbone == 'resnet18':
            self.pretrained = resnet18_v1s(pretrained=pretrained_base, dilated=True, local_rank=local_rank, **kwargs)
        elif backbone == 'resnet50':
            self.pretrained = resnet50_v1s(pretrained=pretrained_base, local_rank=local_rank, dilated=True, **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101_v1s(pretrained=pretrained_base, local_rank=local_rank, dilated=True, **kwargs)
        elif backbone == 'resnet152':
            self.pretrained = resnet152_v1s(pretrained=pretrained_base, local_rank=local_rank, dilated=True, **kwargs)
            # self.pretrained = models.resnet152()
            # # # load pretrained model:
            # model_dict = self.pretrained.state_dict()
            # pretrained_file = model_zoo.load_url(model_urls['resnet152'])
            # pretrained_file = {k: v for k, v in pretrained_file.items() if (k in model_dict)}
            # model_dict.update(pretrained_file)
            # #
            # self.pretrained.load_state_dict(model_dict)
            # #
            # # # remove fully connected layer, avg pool and layer5:
            # self.pretrained = nn.Sequential(*list(self.pretrained.children())[:-3])
            # torchvision.models.resnet152(weights='IMAGENET1K_V1')

        elif backbone == 'resnet18_original':
            self.pretrained = resnet50_v1b(pretrained=pretrained_base, dilated=True, local_rank=local_rank, **kwargs)
        elif backbone == 'resnet50_original':
            self.pretrained = resnet50_v1b(pretrained=pretrained_base, local_rank=local_rank, dilated=True, **kwargs)
        elif backbone == 'resnet101_original':
            self.pretrained = resnet101_v1b(pretrained=pretrained_base, local_rank=local_rank, dilated=True, **kwargs)

        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        
        if self.backbone.split('_')[-1] == 'original':
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
        else:
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu1(x)

            x = self.pretrained.conv2(x)
            x = self.pretrained.bn2(x)
            x = self.pretrained.relu2(x)

            x = self.pretrained.conv3(x)
            x = self.pretrained.bn3(x)
            x = self.pretrained.relu3(x)
            x = self.pretrained.maxpool(x)
        
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred
