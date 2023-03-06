import torch
import torch.nn as nn
from torchinfo import summary

from resnest.torch import resnest101
from pretrainedmodels import se_resnext101_32x4d
import torchvision.models as models
from torchvision.models import EfficientNet_B7_Weights, ResNet152_Weights

class Effnet_Melanoma(nn.Module):
    """It uses efficientnet b7
    to have a good classifier please
    make sure to use images of 600x600"""

    def __init__(self, out_dim):
        super(Effnet_Melanoma, self).__init__()

        # Take the efficient net b7 as base
        self.net = models.efficientnet_b7(
            weights=EfficientNet_B7_Weights.DEFAULT)

        # Define the classifier for the melanoma problem into a separated layer
        in_dim = self.net.classifier.in_features
        self.classifier = nn.Linear(in_dim, out_dim)

        # Disable efficient net b7 classifier
        self.net.classifier = nn.Identity()

        # Definition of multiple dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

    def extract(self, x):
        x = self.net(x)     \
            .squeeze(-1)    \
            .squeeze(-1) # One squeeze could be enough
        return x

    def forward(self, x):
        # Transfer learning here
        x = self.extract(x)

        # Apply multiple dropouts
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.classifier(dropout(x))
            else:
                out += self.classifier(dropout(x))

        # Compute the average of dropouts
        out /= len(self.dropouts)

        return out


class Resnest_Melanoma(nn.Module):
    def __init__(self, out_dim):
        super(Resnest_Melanoma, self).__init__()

        # Take Resnet152 as base
        self.net = models.resnet152(ResNet152_Weights.DEFAULT)

        # Define the classifier for the melanoma problem into a separated layer
        in_dim = self.net.classifier.in_features
        self.classifier = nn.Linear(in_dim, out_dim)

        # Disable efficient net b7 classifier
        self.net.classifier = nn.Identity()

        # Definition of multiple dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

    def extract(self, x):
        x = self.net(x)     \
            .squeeze(-1)    \
            .squeeze(-1) # One squeeze could be enough
        return x

    def forward(self, x):
        # Transfer learning here
        x = self.extract(x)

        # Apply multiple dropouts
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.classifier(dropout(x))
            else:
                out += self.classifier(dropout(x))

        # Compute the average of dropouts
        out /= len(self.dropouts)

        return out


class Seresnext_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, pretrained=False):
        super(Seresnext_Melanoma, self).__init__()
        if pretrained:
            self.net = se_resnext101_32x4d(
                num_classes=1000, pretrained='imagenet')
        else:
            self.net = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        self.net.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.net.last_linear.in_features
        self.classifier = nn.Linear(in_ch, out_dim)
        self.net.last_linear = nn.Identity()

    def extract(self, x):
        x = self.net(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.classifier(dropout(x))
            else:
                out += self.classifier(dropout(x))
        out /= len(self.dropouts)
        return out
