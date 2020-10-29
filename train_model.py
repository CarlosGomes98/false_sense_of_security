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
from trainers import Trainer, FGSMTrainer, CurvatureRegularizationTrainer, GradientRegularizationTrainer
from Nets import CIFAR_Wide_Res_Net, CIFAR_Res_Net
NAME_TO_MODEL = {'wide_res_net': CIFAR_Wide_Res_Net, 'res_net': CIFAR_Res_Net}

def train_model(model_name, strategy, output_path, epochs, eps=None):
    # setup
    device = torch.device("cuda")
    batch_size = 128
    test_batch_size = 1000
    log_interval = 10
    transform = transform = transforms.Compose(
                [transforms.ToTensor()]
    )
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=2)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                    shuffle=False, num_workers=2)
    classes = classes = ('plane', 'car', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    model = NAME_TO_MODEL[model_name](device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if strategy == 'normal':
        trainer = Trainer(device=device, log_interval=log_interval)
        trainer.train(model, train_loader, epochs, test_loader=test_loader, optimizer=optimizer)
    elif strategy == 'fgsm':
        if eps is None:
            raise Exception("Need an epsilon to preform fgsm training")
        trainer = FGSMTrainer(device=device, log_interval=log_interval, clip_min=0, clip_max=1, eps=eps)
        trainer.train(model, train_loader, epochs, test_loader=test_loader, optimizer=optimizer)
    elif strategy == 'gradient_regularization':
        trainer = GradientRegularizationTrainer(device=device, log_interval=log_interval)
        trainer.train(model, train_loader, epochs, test_loader=test_loader, optimizer=optimizer)
    elif strategy == 'curvature_regularization':
        trainer = CurvatureRegularizationTrainer(device=device, log_interval=log_interval, lambda_=4)
        trainer.train(model, train_loader, epochs, test_loader=test_loader, optimizer=optimizer, h=[0.1, 0.4, 0.8, 1.8, 3])
    torch.save(model.state_dict(), output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with a given strategy.')
    # TODO: maybe generalize to other datasets
    parser.add_argument('--model', type=str, choices=['wide_res_net', 'res_net'], default='wide_res_net', help='name of the model')
    parser.add_argument('--strategy', type=str, choices=['normal', 'fgsm', 'curvature_regularization', 'gradient_regularization'], default='normal', help='training strategy')
    parser.add_argument('--eps', type=float, help='eps size for fgsm')
    parser.add_argument('--epochs', type=int, default=10, help='epochs to train for')
    parser.add_argument('--output_path', type=str, default='models\output.model', help='name of the model')

    

    args = parser.parse_args()

    train_model(args.model, args.strategy, args.output_path, args.epochs, eps=args.eps)