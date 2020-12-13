import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, FGSM
from src.trainers import Trainer, FGSMTrainer, CurvatureRegularizationTrainer, GradientRegularizationTrainer, StepllTrainer, PGDTrainer
from src.Nets import CIFAR_Wide_Res_Net, CIFAR_Res_Net, CIFAR_Net

# Main file used to instantiate and train models

NAME_TO_MODEL = {
    'wide_res_net': CIFAR_Wide_Res_Net,
    'res_net': CIFAR_Res_Net,
    'simple': CIFAR_Net
}
'''
Train a given model architecture with a given robustness strategy
Can also save the norm of gradients w.r.t. input at each iteration
'''


def train_model(model_name,
                strategy,
                output_path,
                epochs,
                eps=None,
                report_gradient_norm=False,
                lambda_=None,
                model_path=None):
    # setup
    device = torch.device("cuda")
    batch_size = 128
    test_batch_size = 128
    log_interval = 10
    transform = transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data',
                                     train=True,
                                     download=True,
                                     transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)
    test_dataset = datasets.CIFAR10(root='./data',
                                    train=False,
                                    download=True,
                                    transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              num_workers=2)
    classes = classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                         'horse', 'ship', 'truck')

    if report_gradient_norm:
        report_gradient_norm = output_path.split('.')[0]
    else:
        report_gradient_norm = None

    model = NAME_TO_MODEL[model_name]().to(device)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if strategy == 'normal':
        trainer = Trainer(device=device,
                          log_interval=log_interval,
                          report_gradient_norm=report_gradient_norm)
        trainer.train(model,
                      train_loader,
                      epochs,
                      test_loader=test_loader,
                      optimizer=optimizer)
    elif strategy == 'fgsm':
        if eps is None:
            raise Exception("Need an epsilon to preform fgsm training")
        trainer = FGSMTrainer(device=device,
                              log_interval=log_interval,
                              clip_min=0,
                              clip_max=1,
                              eps=eps,
                              report_gradient_norm=report_gradient_norm)
        trainer.train(model,
                      train_loader,
                      epochs,
                      test_loader=test_loader,
                      optimizer=optimizer)
    elif strategy == 'pgd':
        if eps is None:
            raise Exception("Need an epsilon to preform pgd training")
        trainer = PGDTrainer(device=device,
                              log_interval=log_interval,
                              clip_min=0,
                              clip_max=1,
                              eps=eps,
                              report_gradient_norm=report_gradient_norm)
        trainer.train(model,
                      train_loader,
                      epochs,
                      test_loader=test_loader,
                      optimizer=optimizer)
    elif strategy == 'step_ll':
        if eps is None:
            raise Exception("Need an epsilon to preform step-ll training")
        trainer = StepllTrainer(device=device,
                                log_interval=log_interval,
                                clip_min=0,
                                clip_max=1,
                                eps=eps,
                                report_gradient_norm=report_gradient_norm)
        trainer.train(model,
                      train_loader,
                      epochs,
                      test_loader=test_loader,
                      optimizer=optimizer)
    elif strategy == 'gradient_regularization':
        if lambda_ is None:
            raise Exception(
                "Need a lambda to preform gradient regularization training")
        trainer = GradientRegularizationTrainer(
            device=device,
            log_interval=log_interval,
            report_gradient_norm=report_gradient_norm,
            lambda_=lambda_)
        trainer.train(model,
                      train_loader,
                      epochs,
                      test_loader=test_loader,
                      optimizer=optimizer)
    elif strategy == 'curvature_regularization':
        trainer = CurvatureRegularizationTrainer(
            device=device,
            log_interval=log_interval,
            lambda_=4,
            report_gradient_norm=report_gradient_norm)
        trainer.train(model,
                      train_loader,
                      epochs,
                      test_loader=test_loader,
                      optimizer=optimizer,
                      h=[0.1, 0.4, 0.8, 1.8, 3])
    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a model with a given strategy.')
    # TODO: maybe generalize to other datasets
    parser.add_argument('--model',
                        type=str,
                        choices=['simple', 'wide_res_net', 'res_net'],
                        default='res_net',
                        help='name of the model')
    parser.add_argument('--strategy',
                        type=str,
                        choices=[
                            'normal', 'fgsm', 'step_ll',
                            'curvature_regularization',
                            'gradient_regularization',
                            'pgd'
                        ],
                        default='normal',
                        help='training strategy')
    parser.add_argument('--eps', type=float, help='eps size for fgsm')
    parser.add_argument(
        '--lambda_',
        type=float,
        help='lambda for training with gradient regularization')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='epochs to train for')
    parser.add_argument('--output_path',
                        type=str,
                        default='models\output.model',
                        help='name of the model')
    parser.add_argument(
        '--report_gradient_norm',
        action='store_true',
        help='Will save gradient norms to directory at output path')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Allows a model to be provided, which will be trained on')

    args = parser.parse_args()

    train_model(args.model,
                args.strategy,
                args.output_path,
                args.epochs,
                eps=args.eps,
                report_gradient_norm=args.report_gradient_norm,
                lambda_=args.lambda_,
                model_path=args.model_path)
