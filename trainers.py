import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from utils import fgsm_

class Trainer:
    def __init__(self, device="cpu", log_interval=10):
        self.device = device
        self.log_interval = log_interval
        
    def train(self, model, train_loader, epochs, test_loader=None, optimizer=None):
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, epochs + 1):
            self.train_step(model, train_loader, epoch, optimizer, criterion)
            if test_loader is not None:
                self.test(model, test_loader, criterion)
    
    def test(self, model, test_loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += criterion(output, target).sum().item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        
    def train_step(self, model, train_loader, epoch, optimizer, criterion):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                
class FGSMTrainer(Trainer):
    def __init__(self, device="cpu", log_interval=10, clip_min=0, clip_max=1, eps=(8/255)):
        super(FGSMTrainer, self).__init__(device=device, log_interval=log_interval)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps = eps
        
    def train(self, model, train_loader, epochs, test_loader=None, optimizer=None):
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, epochs + 1):
            self.train_step(model, train_loader, epoch, optimizer, criterion)
            if test_loader is not None:
                self.test(model, test_loader, criterion)

    def train_step(self, model, train_loader, epoch, optimizer, criterion):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            adv_data = fgsm_(model, data, target, eps=self.eps, targeted=False, device=self.device, clip_min=self.clip_min, clip_max=self.clip_max)
            optimizer.zero_grad()
            output = model(adv_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))