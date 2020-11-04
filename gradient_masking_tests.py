import argparse
import tqdm
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
from advertorch.attacks import LinfSPSAAttack
from trainers import Trainer, FGSMTrainer
from robustbench.model_zoo.models import Carmon2019UnlabeledNet
from utils import adversarial_accuracy, fgsm_
import eagerpy as ep
from Nets import CIFAR_Wide_Res_Net, CIFAR_Res_Net, CIFAR_Net

def run_masking_benchmarks(model, test_dataset, epsilon=0.06, device="cpu", batch_size=32):
    """
    This method runs through a checklist of potential indicators of gradient masking, as exposed in 
    "Obfuscated Gradients Give a False Sense of Security:
    Circumventing Defenses to Adversarial Examples"
    https://arxiv.org/pdf/1802.00420.pdf
    """
    acc = get_accuracy(model, test_dataset, epsilon=epsilon, device=device, batch_size=batch_size)*100
    fgsm_acc_small = get_accuracy(model, test_dataset, epsilon=epsilon/10, device=device, batch_size=batch_size, attack=FGSM())*100
    fgsm_acc_med = get_accuracy(model, test_dataset, epsilon=epsilon/2, device=device, batch_size=batch_size, attack=FGSM())*100
    fgsm_acc = get_accuracy(model, test_dataset, epsilon=epsilon, device=device, batch_size=batch_size, attack=FGSM())*100
    pgd_acc = get_accuracy(model, test_dataset, epsilon=epsilon, device=device, batch_size=batch_size, attack=LinfPGD(steps=20, rel_stepsize=1/8))*100
    pgd_unbounded = get_accuracy(model, test_dataset, epsilon=1, device=device, batch_size=batch_size, attack=LinfPGD(steps=20, rel_stepsize=1/8))*100
    spsa_acc = spsa_accuracy(model, test_dataset, eps=epsilon, iters=10, nb_sample=128, batch_size=8, device=device)*100
    
    print("Model accuracy: {}%".format(acc))
    print("FGSM attack model accuracy -- eps = {}: {}%, eps = {}: {}%, eps = {}: {}%".format(epsilon/10, fgsm_acc_small, epsilon/2, fgsm_acc_med, epsilon, fgsm_acc))
    if not (fgsm_acc < fgsm_acc_med and fgsm_acc_med < fgsm_acc_small):
        print("Gradient Masking Warning: Increasing epsilon did not improve the attack!!")
    

    print("PGD accuracy: {}%".format(pgd_acc))
    if (pgd_acc > fgsm_acc):
        print("Gradient Masking Warning: PGD attack was not stronger than FGSM attack!!")
    
    print("Unbounded PGD model accuracy: {}%".format(pgd_unbounded))

    print("SPSA accuracy: {}%".format(spsa_acc))

    if spsa_acc < fgsm_acc:
        print("Gradient Masking Warning: Black Box attack was stronger than FGSM attack!!")
    
    if spsa_acc < pgd_acc:
        print("Gradient Masking Warning: Black Box attack was stronger than PGD attack!!")

def get_accuracy(model, test_dataset, attack=None, epsilon=0.03, subset_size=10000, device="cpu", batch_size=32):
    """
    Reports the accuracy of the model, potentially under some attack (e.g. FGSM, PGD, ...)
    """
    fmodel = PyTorchModel(model, bounds=(0, 1))
    correct = 0
    subset = torch.utils.data.Subset(test_dataset, np.random.randint(0, len(test_dataset), size=subset_size).tolist())
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                 shuffle=False, num_workers=2)
    for images, labels in tqdm.tqdm(subset_loader):
        images, labels = images.to(device), labels.type(torch.cuda.LongTensor)
        if attack is None:
            correct += accuracy(fmodel, images, labels) * images.shape[0]
        else:
            _, _, success = attack(fmodel, images, labels, epsilons=epsilon)
            correct += (~success).sum().item()
    return correct / subset_size

def spsa_accuracy(model, test_dataset, eps=0.03, iters=1, nb_sample=128, batch_size=8, device="cpu", subset_size=100):
    """
    Reports the accuracy of the model under the SPSA attack. This method is quite expensive, so a small subset_size is reccomended,
    particularly for deeper networks.
    """
    attack = LinfSPSAAttack(model, eps, nb_iter=iters, nb_sample=nb_sample, loss_fn=nn.CrossEntropyLoss(reduction='none'))
    subset = torch.utils.data.Subset(test_dataset, np.random.randint(0, len(test_dataset), size=subset_size).tolist())
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                 shuffle=False, num_workers=2)
    correct = 0
    for images, labels in tqdm.tqdm(subset_loader):
        images, labels = images.to(device), labels.type(torch.cuda.LongTensor)
        adv = attack.perturb(images, labels)
        preds = model(adv).argmax(-1)
        correct += (preds == labels).sum().item()
    return correct / len(subset_loader.dataset)


