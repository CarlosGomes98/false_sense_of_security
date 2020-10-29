import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import grad
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
        model.train()
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

# FROM https://github.com/F-Salehi/CURE_robustness
# "Robustness via curvature regularization, and vice versa ", SM. Moosavi-Dezfooli, A. Fawzi, J. Uesato, and P. Frossard, CVPR 2019.
class GradientRegularizationTrainer(Trainer):
    def __init__(self, device="cpu", log_interval=10, lambda_=0.1):
        super(GradientRegularizationTrainer, self).__init__(device=device, log_interval=log_interval)
        self.lambda_ = lambda_

    def train_step(self, model, train_loader, epoch, optimizer, criterion):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad_()
            output = model(data)
            loss = criterion(output, target)
            gradient_norm = (grad(outputs=loss, inputs=data, retain_graph=True, only_inputs=True)[0]**2).sum()
            loss = loss + self.lambda_ * gradient_norm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

# FROM https://github.com/F-Salehi/CURE_robustness
# "Robustness via curvature regularization, and vice versa ", SM. Moosavi-Dezfooli, A. Fawzi, J. Uesato, and P. Frossard, CVPR 2019.
class CurvatureRegularizationTrainer(Trainer):
    def __init__(self, device="cpu", log_interval=10, lambda_=4):
        super(CurvatureRegularizationTrainer, self).__init__(device=device, log_interval=log_interval)
        self.lambda_ = lambda_

    def _find_z(self, model, criterion, inputs, targets, h):
        '''
        Finding the direction in the regularizer
        '''
        inputs.requires_grad_()
        outputs = model.eval()(inputs)
        loss_z = criterion(model.eval()(inputs), targets)                
        loss_z.backward(torch.ones(targets.size()).to(self.device))         
        grad = inputs.grad.data + 0.0
        norm_grad = grad.norm().item()
        z = torch.sign(grad).detach() + 0.
        z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)  
        zero_gradients(inputs) 
        model.zero_grad()

        return z, norm_grad
    
    def regularizer(self, model, criterion, inputs, targets, h = 3.):
        '''
        Regularizer term in CURE
        '''
        z, norm_grad = self._find_z(model, criterion, inputs, targets, h)
        
        inputs.requires_grad_()
        outputs_pos = model.eval()(inputs + z)
        outputs_orig = model.eval()(inputs)

        loss_pos = criterion(outputs_pos, targets)
        loss_orig = criterion(outputs_orig, targets)
        grad_diff = torch.autograd.grad((loss_pos-loss_orig), inputs, grad_outputs=torch.ones(targets.size()).to(self.device),
                                        create_graph=True)[0]
        reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
        model.zero_grad()

        return torch.sum(self.lambda_ * reg) / float(inputs.size(0)), norm_grad
    
    def train(self, model, train_loader, epochs, test_loader=None, optimizer=None, h=[0.1, 0.4, 0.8, 1.8, 3]):
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, epochs + 1):
            self.train_step(model, train_loader, epoch, optimizer, criterion, h=h)
            if test_loader is not None:
                self.test(model, test_loader, criterion)

    def train_step(self, model, train_loader, epoch, optimizer, criterion, h=[0.1, 0.4, 0.8, 1.8, 3]):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            regularizer, grad_norm = self.regularizer(model, criterion, data, target, h=h)
            output = model(data)
            loss = criterion(output, target) + regularizer
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))