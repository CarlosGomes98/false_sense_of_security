import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from src.utils import fgsm_, pgd_
from robustbench.model_zoo.cifar10 import Carmon2019UnlabeledNet
from robustbench.model_zoo.architectures.resnet import ResNet, BasicBlock
from torch.nn.modules.utils import _pair
from .Nets.CURE import CUREResNet, CUREBasicBlock
from .Nets.STEP import resnet as StepResNet


# Most models taken from robustbench model zoo, to ensure comparability with literature
class Normalization(nn.Module):
    """
    Torch layer that handles normalization of data.

    This normalization class ensures the attacks do not need to know about 
    the preprocessing steps done on the data, and therefore step size can
    remain unchaged.
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor(mean).view((1, 3, 1, 1))
        self.sigma = torch.FloatTensor(std).view((1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.sigma.to(x.device)


# ResNet 18
class CIFAR_Res_Net(ResNet):
    """
    ResNet 18 from robustbench zoo extended with a normalization layer at the beggining.
    """
    def __init__(self, normalization_mean=[0.5, 0.5, 0.5], normalization_std=[0.5, 0.5, 0.5], layers=[2, 2, 2, 2]):
        super(CIFAR_Res_Net, self).__init__(BasicBlock, layers)
        self.norm = Normalization(normalization_mean, normalization_std)


    def forward(self, x):
        x = self.norm(x)
        return super().forward(x)


class CIFAR_Wide_Res_Net(Carmon2019UnlabeledNet):
    """
    WideResNet-28-10 from robustbench zoo(Carmon2019UnlabeledNet) extended with a normalization layer at the beggining.
    """
    def __init__(self):
        super(CIFAR_Wide_Res_Net, self).__init__()
        self.norm = Normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


    def forward(self, x):
        x = self.norm(x)
        return super().forward(x)


class CIFAR_Net(nn.Module):
    """
    Very simple Conv Net for CIFAR-10.
    """
    def __init__(self):
        super(CIFAR_Net, self).__init__()
        self.norm = Normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv1_1 = nn.Conv2d(32, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2_1 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv1_1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv2_1(x)))
        x = x.reshape(-1, 64 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def CUREResNet18():
    return nn.Sequential(Normalization([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]), CUREResNet(CUREBasicBlock, [2,2,2,2]))

def StepResNet18():
    return nn.Sequential(Normalization([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]), StepResNet('resnet18'))