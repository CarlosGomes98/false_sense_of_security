import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from src.utils import fgsm_, pgd_
from robustbench.model_zoo.models import Carmon2019UnlabeledNet
from robustbench.model_zoo.resnet import ResNet, BasicBlock
from . import blocks
from .blocks import Conv, Linear
from torch.nn.modules.utils import _pair


# Most models taken from robustbench model zoo, to ensure comparability with literature
class Normalization(nn.Module):
    """
    Torch layer that handles normalization of data.

    This normalization class ensures the attacks do not need to know about 
    the preprocessing steps done on the data, and therefore step size can
    remain unchaged.
    """
    def __init__(self, device, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([mean]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([std]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma


# ResNet 18
class CIFAR_Res_Net(ResNet):
    """
    ResNet 18 from robustbench zoo extended with a normalization layer at the beggining.
    """
    def __init__(self, device):
        super(CIFAR_Res_Net, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.norm = Normalization(device, 0.5, 0.5)

        self.to(device)

    def forward(self, x):
        x = self.norm(x)
        return super().forward(x)


class CIFAR_Wide_Res_Net(Carmon2019UnlabeledNet):
    """
    WideResNet-28-10 from robustbench zoo(Carmon2019UnlabeledNet) extended with a normalization layer at the beggining.
    """
    def __init__(self, device):
        super(CIFAR_Wide_Res_Net, self).__init__()
        self.norm = Normalization(device, 0.5, 0.5)

        self.to(device)

    def forward(self, x):
        x = self.norm(x)
        return super().forward(x)


class CIFAR_Net(nn.Module):
    """
    Very simple Conv Net for CIFAR-10.
    """
    def __init__(self, device="cpu"):
        super(CIFAR_Net, self).__init__()
        self.norm = Normalization(device, 0.5, 0.5)
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv1_1 = nn.Conv2d(32, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2_1 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        self.device = device
        self.to(device)

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


class ResNetGradientRegularization(nn.Module):
    """
    Model used in "SCALEABLE INPUT GRADIENT REGULARIZATION FOR ADVERSARIAL ROBUSTNESS"
    Just for loading trained model. Training done using paper's github
    """
    def __init__(self,
                 layers,
                 block='BasicBlock',
                 in_channels=3,
                 classes=10,
                 kernel_size=(3, 3),
                 conv0_kwargs={
                     'kernel_size': (3, 3),
                     'stride': 1
                 },
                 conv0_pool=None,
                 downsample_pool=nn.AvgPool2d,
                 base_channels=16,
                 nonlinear='relu'):
        super().__init__()
        kernel_size = _pair(kernel_size)

        def make_layer(n, block, in_channels, out_channels, stride):
            sublayers = []
            if not in_channels == out_channels:
                sublayers.append(
                    Conv(in_channels,
                         out_channels,
                         kernel_size=(1, 1),
                         nonlinear=nonlinear))

            if stride > 1:
                sublayers.append(downsample_pool(stride))

            for k in range(n):
                sublayers.append(
                    block(out_channels,
                          kernel_size=kernel_size,
                          nonlinear=nonlinear))

            return nn.Sequential(*sublayers)

        block = getattr(blocks, block)

        self.normalization = Normalization(torch.device("cuda"), 0.5, 0.5)
        self.layer0 = Conv(in_channels,
                           base_channels,
                           **conv0_kwargs,
                           nonlinear=nonlinear)
        if conv0_pool:
            self.maxpool = conv0_pool
        else:
            self.maxpool = False

        _layers = []
        for i, n in enumerate(layers):

            if i == 0:
                _layers.append(
                    make_layer(n, block, base_channels, base_channels, 1))
            else:
                _layers.append(
                    make_layer(n, block, base_channels * (2**(i - 1)),
                               base_channels * (2**i), 2))

        self.layers = nn.Sequential(*_layers)

        self.pool = Avg2d()
        self.view = View((2**i) * base_channels)

        self.fc = Linear((2**i) * base_channels, classes, nonlinear=nonlinear)

    @property
    def num_parameters(self):
        return sum([w.numel() for w in self.parameters()])

    def forward(self, x):
        x = self.normalization(x)
        x = self.layer0(x)
        if self.maxpool:
            x = self.maxpool(x)
        x = self.layers(x)
        x = self.pool(x)
        x = self.view(x)
        x = self.fc(x)

        return x


class View(nn.Module):
    def __init__(self, o):
        super(View, self).__init__()
        self.o = o

    def forward(self, x):
        return x.view(-1, self.o)


class Avg2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        sh = x.shape
        x = x.contiguous().view(sh[0], sh[1], -1)
        return x.mean(-1)


def ResNet18GradReg(**kwargs):
    m = ResNetGradientRegularization([2, 2, 2, 2],
                                     block='BasicBlock',
                                     base_channels=32,
                                     **kwargs)
    return m


# OLD DEPRECATED BELOW:
# dont want to remove them yet as might be useful to look at
class MNIST_Net(nn.Module):
    def __init__(self,
                 device="cpu",
                 log_interval=100,
                 batch_size=64,
                 test_batch_size=1000,
                 oracle=None,
                 binary=False):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        if binary:
            self.fc2 = nn.Linear(128, 2)
        else:
            self.fc2 = nn.Linear(128, 10)
        self.device = device
        self.log_interval = log_interval
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.normalized_min = (0 - 0.1307) / 0.3081
        self.normalized_max = (1 - 0.1307) / 0.3081
        self.train_dataset = datasets.MNIST('data',
                                            train=True,
                                            download=True,
                                            transform=transform)
        self.test_dataset = datasets.MNIST('data',
                                           train=True,
                                           download=True,
                                           transform=transform)

        if binary:
            mask = self.train_dataset.targets < 2
            self.train_dataset.data = self.train_dataset.data[mask]
            self.train_dataset.targets = self.train_dataset.targets[mask]

            mask = self.test_dataset.targets < 2
            self.test_dataset.data = self.test_dataset.data[mask]
            self.test_dataset.targets = self.test_dataset.targets[mask]

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=batch_size,
                                                        num_workers=2,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=test_batch_size,
            num_workers=2,
            shuffle=False)
        self.oracle = oracle
        self.binary = binary
        self.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def train_on_data(self, epochs):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch, optimizer, criterion)
            self.test(criterion)

    def test(self, criterion):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                test_loss += criterion(
                    output, target).sum().item()  # sum up batch loss
                pred = output.argmax(
                    dim=1,
                    keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(test_loss, correct, len(self.test_loader.dataset),
                     100. * correct / len(self.test_loader.dataset)))

    def train_epoch(self, epoch, optimizer, criterion):
        self.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            if self.oracle is not None:
                with torch.no_grad():
                    target = self.oracle(data).argmax(axis=1).type(
                        torch.cuda.LongTensor)
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))


class Gradient_Masked_MNIST(MNIST_Net):
    def __init__(self,
                 device="cpu",
                 log_interval=100,
                 batch_size=64,
                 test_batch_size=1000,
                 binary=False,
                 eps=0.5):
        super(Gradient_Masked_MNIST,
              self).__init__(device=device,
                             log_interval=log_interval,
                             batch_size=batch_size,
                             test_batch_size=test_batch_size,
                             binary=binary)
        self.eps = eps

    def train_epoch(self, epoch, optimizer, criterion):
        # adversarially train using large FGSM single step
        self.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            adv_data = fgsm_(self,
                             data,
                             target,
                             eps=self.eps,
                             targeted=False,
                             device=self.device,
                             clip_min=self.normalized_min,
                             clip_max=self.normalized_max)
            optimizer.zero_grad()
            output = self(adv_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))


class PGD_MNIST(MNIST_Net):
    def __init__(self,
                 device="cpu",
                 log_interval=100,
                 batch_size=64,
                 test_batch_size=1000,
                 step=0.1,
                 eps=0.7,
                 iters=7,
                 binary=False):
        super(PGD_MNIST, self).__init__(device=device,
                                        log_interval=log_interval,
                                        batch_size=batch_size,
                                        test_batch_size=test_batch_size,
                                        binary=binary)
        self.step = step
        self.eps = eps
        self.iters = iters

    def train_epoch(self, epoch, optimizer, criterion):
        # adversarially train using PGD
        self.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            adv_data = pgd_(self,
                            data,
                            target,
                            step=self.step,
                            eps=self.eps,
                            iters=self.iters,
                            targeted=False,
                            device=self.device,
                            clip_min=self.normalized_min,
                            clip_max=self.normalized_max)
            optimizer.zero_grad()
            output = self(adv_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))