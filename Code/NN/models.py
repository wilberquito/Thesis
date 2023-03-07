"""
Model classes where transfer learning is used.
I use already trainned as base of image detection.
"""
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (ConvNeXt_Base_Weights, EfficientNet_B7_Weights,
                                ResNet152_Weights)


class Effnet_Melanoma(nn.Module):

    """It uses efficientnet b7 to have a good classifier please make sure to use images of 600x600"""

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
            .squeeze(-1)  # One squeeze could be enough
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

    """It uses resnet152 to have a good classifier please make sure to use images of 232x232"""

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
            .squeeze(-1)  # One squeeze could be enough
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


class ConvNext_Melanoma(nn.Module):

    """It uses convnext base to have a good classifier please make sure to use images of 232x232"""

    def __init__(self, out_dim):
        super(ConvNext_Melanoma, self).__init__()

        # Take ConvNext Base as base
        self.net = models.convnext_base(ConvNeXt_Base_Weights.DEFAULT)

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
            .squeeze(-1)  # One squeeze could be enough
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
