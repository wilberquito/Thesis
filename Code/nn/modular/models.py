"""
Model classes where transfer learning is used.
I use already trainned as base of image detection.
"""
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (ConvNeXt_Base_Weights, EfficientNet_B7_Weights,
                                ResNet152_Weights)

from .utility import model_input_size


class BaseMelanoma(nn.Module):

    def __init__(self):
        super().__init__()


    def extract(self, x):
        x = self.net(x)     \
            .squeeze(-1)    \
            .squeeze(-1)  # One squeeze could be enough
        return x


    def freeze(self):
        # Don't compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = False


class Effnet_Melanoma(BaseMelanoma):
    """It uses efficientnet b7 to have a good classifier please make sure to use images of 600x600"""

    def __init__(self, out_dim):
        super(Effnet_Melanoma, self).__init__()

        # Take the efficient net b7 as base
        self.net = models.efficientnet_b7(
            weights=EfficientNet_B7_Weights.DEFAULT)

        # Take the input of the fully connected layer of effnet
        in_dim = self.net.classifier[-1].in_features

        # Disable efficient net b7 classifier
        self.net.classifier = nn.Identity()

        # Declare the fully connected layer
        self.classifier = nn.Linear(in_dim, out_dim)

        # Definition of multiple dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

        # Freeze base layers
        self.freeze()


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


class Resnest_Melanoma(BaseMelanoma):
    """It uses resnet152 to have a good classifier please make sure to use images of 232x232"""

    def __init__(self, out_dim):
        super(Resnest_Melanoma, self).__init__()

        # Take Resnet152 as base
        self.net = models.resnet152(weights=ResNet152_Weights.DEFAULT)

        # Take the input of the fully connected layer of resnet
        in_dim = self.net.fc.in_features

        # Disable efficient net b7 classifier
        self.net.fc = nn.Identity()

        # Declare the fully connected layer
        self.classifier = nn.Linear(in_dim, out_dim)

        # Definition of multiple dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

        # Freeze base layers
        self.freeze()


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


class ConvNext_Melanoma(BaseMelanoma):
    """It uses convnext base to have a good classifier please make sure to use images of 232x232"""

    def __init__(self, out_dim):
        super(ConvNext_Melanoma, self).__init__()

        # Take ConvNext Base as base
        self.net = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)

        # Take the input of the fully connected layer of convnext
        in_dim = self.net.classifier[-1].in_features

        # Disable conv next net classifier
        self.net.classifier = nn.Identity()

        # Declare the fully connected layer
        self.classifier = nn.Linear(in_dim, out_dim)

        # Definition of multiple dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

        # Freeze base layers
        self.freeze()


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


def get_input_size(model_name):
    """Ask which is the recomended input size for the supported models"""
    assert model_name in ['effnet', 'resnet', 'convnext']
    return model_input_size(model_name)