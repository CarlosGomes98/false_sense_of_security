import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import grad
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from src.utils import fgsm_, step_ll_, pgd_
from src.gradient_masking_tests import gradient_norm

lr = 0.001

class Trainer:
    """
    Base class that trains a given model, given a dataloader using standard SGD.

    All other trainer classes are extended from this one.
    """
    def __init__(self,
                 device="cpu",
                 log_interval=10,
                 report_gradient_norm=None):
        self.device = device
        self.log_interval = log_interval
        self.report_gradient_norm = report_gradient_norm
        if report_gradient_norm is not None:
            if os.path.isdir(report_gradient_norm):
                raise Exception(
                    "Folder with gradient norm information already exists. Would overwrite existing log."
                )
            os.mkdir(report_gradient_norm)

    def train(self,
              model,
              train_loader,
              epochs,
              test_loader=None,
              optimizer=None):
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, epochs + 1):
            self.train_step(model, train_loader, epoch, optimizer, criterion)
            if test_loader is not None:
                self.test(model, test_loader, criterion)
            if self.report_gradient_norm is not None:
                norm = gradient_norm(model,
                                     train_loader,
                                     device=self.device,
                                     subset_size=5000)
                torch.save(
                    norm,
                    os.path.join(self.report_gradient_norm,
                                 'epoch_{}.pt'.format(epoch)))
                print('Gradient Norm -- Mean: {}, Min: {}, Max: {}'.format(
                    norm.mean(), norm.min(), norm.max()))

    def test(self, model, test_loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += criterion(
                    output, target).sum().item()  # sum up batch loss
                pred = output.argmax(
                    dim=1,
                    keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(test_loss, correct, len(test_loader.dataset),
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
    """
    Extends base trainer class to implement training with a single FGSM step.
    """
    def __init__(self,
                 device="cpu",
                 log_interval=10,
                 clip_min=0,
                 clip_max=1,
                 eps=(8 / 255),
                 report_gradient_norm=None):
        super(FGSMTrainer,
              self).__init__(device=device,
                             log_interval=log_interval,
                             report_gradient_norm=report_gradient_norm)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps = eps

    def train_step(self, model, train_loader, epoch, optimizer, criterion):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            adv_data = fgsm_(model,
                             data,
                             target,
                             eps=self.eps,
                             targeted=False,
                             device=self.device,
                             clip_min=self.clip_min,
                             clip_max=self.clip_max)
            optimizer.zero_grad()
            output = model(adv_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                
class PGDTrainer(Trainer):
    """
    Extends base trainer class to implement training with a single FGSM step.
    """
    def __init__(self,
                 device="cpu",
                 log_interval=10,
                 clip_min=0,
                 clip_max=1,
                 eps=(8 / 255),
                 report_gradient_norm=None):
        super(PGDTrainer,
              self).__init__(device=device,
                             log_interval=log_interval,
                             report_gradient_norm=report_gradient_norm)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps = eps

    def train_step(self, model, train_loader, epoch, optimizer, criterion):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            adv_data = pgd_(model,
                             data,
                             target,
                             step=1/4,
                             eps=self.eps,
                             targeted=False,
                             device=self.device,
                             iters=7,
                             random_step=True,
                             clip_min=self.clip_min,
                             clip_max=self.clip_max)
            optimizer.zero_grad()
            output = model(adv_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))



class StepllTrainer(Trainer):
    """
    Extends base trainer class to implement training with a single FGSM step.
    """
    def __init__(self,
                 device="cpu",
                 log_interval=10,
                 clip_min=0,
                 clip_max=1,
                 eps=(8 / 255),
                 report_gradient_norm=None):
        super(StepllTrainer,
              self).__init__(device=device,
                             log_interval=log_interval,
                             report_gradient_norm=report_gradient_norm)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps = eps

    def train_step(self, model, train_loader, epoch, optimizer, criterion):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            adv_data = step_ll_(model,
                                data,
                                target,
                                eps=self.eps,
                                device=self.device,
                                clip_min=self.clip_min,
                                clip_max=self.clip_max)
            optimizer.zero_grad()
            output = model(adv_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

class GradientRegularizationTrainer(Trainer):
    """
    Extends the base trainer class to implement training with simple input gradient regularization w.r.t. the loss.

    """

    def __init__(self,
                 device="cpu",
                 log_interval=10,
                 lambda_=0.1,
                 annealing=False,
                 report_gradient_norm=None):
        super(GradientRegularizationTrainer,
              self).__init__(device=device,
                             log_interval=log_interval,
                             report_gradient_norm=report_gradient_norm)
        self.lambda_ = lambda_
        self.cur_lambda = lambda_
        self.annealing = annealing

    def train(self,
              model,
              train_loader,
              epochs,
              test_loader=None,
              optimizer=None):
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, epochs + 1):
            self.train_step(model, train_loader, epoch, optimizer, criterion)
            if test_loader is not None:
                self.test(model, test_loader, criterion)
            if self.annealing:
                self.cur_lambda = self.lambda_ * ((epochs - epoch) / epochs)
        if self.report_gradient_norm is not None:
            norm = gradient_norm(model,
                                 train_loader,
                                 device=self.device,
                                 subset_size=5000)
            torch.save(
                norm,
                os.path.join(self.report_gradient_norm,
                             'epoch_{}.pt'.format(epoch)))
            print('Gradient Norm -- Mean: {}, Min: {}, Max: {}'.format(
                norm.mean(), norm.min(), norm.max()))

    def train_step(self, model, train_loader, epoch, optimizer, criterion):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad_()
            output = model(data)
            loss = criterion(output, target)
            ce_loss = loss
            # simple input regularization
            input_grad = grad(outputs=loss,
                                inputs=data,
                                create_graph=True,
                                only_inputs=True)[0]
            norm = torch.linalg.norm(input_grad.view(data.shape[0], -1), dim=1)
            
            loss = loss + self.cur_lambda * norm.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\t CE Loss: {:.6f}, Gradient loss: {:.6f}'
                    .format(epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            ce_loss.item(),
                            (self.cur_lambda * norm.sum().item())))

class JacobianRegularizationTrainer(Trainer):
    """
    Extends the base trainer class to implement training with input gradient regularization.
    From: https://arxiv.org/pdf/1803.08680.pdf

    """
    def __init__(self,
                 device="cpu",
                 log_interval=10,
                 lambda_=0.1,
                 annealing=False,
                 report_gradient_norm=None):
        super(JacobianRegularizationTrainer,
              self).__init__(device=device,
                             log_interval=log_interval,
                             report_gradient_norm=report_gradient_norm)
        self.lambda_ = lambda_
        self.cur_lambda = lambda_
        self.annealing = annealing

    def train(self,
              model,
              train_loader,
              epochs,
              test_loader=None,
              optimizer=None):
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, epochs + 1):
            self.train_step(model, train_loader, epoch, optimizer, criterion)
            if test_loader is not None:
                self.test(model, test_loader, criterion)
            if self.annealing:
                self.cur_lambda = self.lambda_ * ((epochs - epoch) / epochs)
        if self.report_gradient_norm is not None:
            norm = gradient_norm(model,
                                 train_loader,
                                 device=self.device,
                                 subset_size=5000)
            torch.save(
                norm,
                os.path.join(self.report_gradient_norm,
                             'epoch_{}.pt'.format(epoch)))
            print('Gradient Norm -- Mean: {}, Min: {}, Max: {}'.format(
                norm.mean(), norm.min(), norm.max()))

    def train_step(self, model, train_loader, epoch, optimizer, criterion):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad_()
            output = model(data)
            loss = criterion(output, target)
            ce_loss = loss
            # simple input regularization
            # input_grad = grad(outputs=loss,
            #                     inputs=data,
            #                     create_graph=True,
            #                     only_inputs=True)[0]
            # norm = torch.norm(input_grad.view(data.shape[0], -1), dim=1)
            # print(norm.sum())

            # to get the jacobian, we need to go through each logit class one by one
            norms = torch.zeros(data.shape[0]).to(self.device)
            for i in range(10):
                model.zero_grad()
                logit = output[:, i]
                gradient = torch.autograd.grad(outputs=logit, inputs=data, grad_outputs=torch.ones_like(logit), only_inputs=True, create_graph=True)[0]
                gradient = gradient.view(data.shape[0], -1)
                norms += torch.linalg.norm(gradient, dim=1)**2
            
            loss = loss + self.cur_lambda * norms.sqrt().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\t CE Loss: {:.6f}, Gradient loss: {:.6f}'
                    .format(epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            ce_loss.item(),
                            (self.cur_lambda * norms.sum().sqrt()).item()))
