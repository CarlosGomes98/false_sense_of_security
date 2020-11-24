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
from robustbench.model_zoo.models import Carmon2019UnlabeledNet
from src.utils import adversarial_accuracy, fgsm_, random_step_
import eagerpy as ep
from src.Nets import CIFAR_Wide_Res_Net, CIFAR_Res_Net, CIFAR_Net

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
    pgd_acc = get_accuracy(model, test_dataset, epsilon=epsilon, device=device, batch_size=batch_size, attack=LinfPGD(steps=7, rel_stepsize=1/4))*100
    pgd_unbounded = get_accuracy(model, test_dataset, epsilon=1, device=device, batch_size=batch_size, attack=LinfPGD(steps=7, rel_stepsize=1/4))*100
    spsa_acc = spsa_accuracy(model, test_dataset, eps=epsilon, iters=10, nb_sample=128, batch_size=8, device=device)*100
    random_acc_small = get_random_accuracy(model, test_dataset, epsilon=epsilon/10, device=device, batch_size=batch_size)*100
    random_acc_med = get_random_accuracy(model, test_dataset, epsilon=epsilon/2, device=device, batch_size=batch_size)*100
    random_acc = get_random_accuracy(model, test_dataset, epsilon=epsilon, device=device, batch_size=batch_size)*100
    
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
    
    print("Random attack model accuracy -- eps = {}: {}%, eps = {}: {}%, eps = {}: {}%".format(epsilon/10, random_acc_small, epsilon/2, random_acc_med, epsilon, random_acc))

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

def get_random_accuracy(model, test_dataset, epsilon=0.03, device="cpu", batch_size=128, subset_size=10000):
    '''
    Calculate the accuracy of the model when subjected to a random attack.
    '''
    correct = 0
    subset = torch.utils.data.Subset(test_dataset, np.random.randint(0, len(test_dataset), size=subset_size).tolist())
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                 shuffle=False, num_workers=2)
    for images, labels in tqdm.tqdm(subset_loader):
        images, labels = images.to(device), labels.type(torch.cuda.LongTensor)
        adv = random_step_(model, images, eps=epsilon, device=device, clip_min=0, clip_max=1)
        preds = model(adv).argmax(-1)
        correct += (preds == labels).sum().item()
    return correct / len(subset_loader.dataset)

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


def gradient_norm(model, data_loader, device='cpu', subset_size=10000):
    """
    Computes the gradient norm w.r.t. the loss at the given points.

    TODO: Move to metrics.
    """
    count = 0
    grad_norms = []
    for (data, target) in data_loader:
        if count > subset_size:
            break
        count += data.shape[0]
        input_ = data.clone().detach_().to(device)
        input_.requires_grad_()
        target = target.to(device)
        model.zero_grad()
        logits = model(input_)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        grad = input_.grad.reshape(input_.shape[0], -1)
        grad_norm = torch.norm(grad, p=2, dim=1)
        grad_norms.append(grad_norm)
    grad_norm = torch.cat(grad_norms)
    return grad_norm


def fgsm_pgd_cos_dif(model, test_dataset, epsilon=0.03, subset_size=1000, device="cpu", batch_size=32, n_steps_pgd=7, return_adjusted_fgsm=True):
    '''
    Method that evaluates how informative the gradients of the network are. Preforms pgd and fgsm and compares the solutions.
    Returns the cosine difference and euclidian distance between the solutions.
    Furthermore, the method computes and returns the success of the adjusted fgsm attack. It takes the output of the fgsm attack
    and rescales it to have the same norm as the pgd solution. This was implemented as it was noticed that the cosine similarity
    was often very close to 1, yet the norm was quite different.
    '''
    fmodel = PyTorchModel(model, bounds=(0, 1))
    cos_dif = []
    distance = []
    successes_fgsm = []
    successes_pgd = []
    if return_adjusted_fgsm:
        successes_adjusted_fgsm = []
    subset = torch.utils.data.Subset(test_dataset, np.random.randint(0, len(test_dataset), size=subset_size).tolist())
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                 shuffle=False, num_workers=2)
    for images, labels in tqdm.tqdm(subset_loader):
        images, labels = images.to(device), labels.type(torch.cuda.LongTensor)
        _, advs_fgsm, success_fgsm = FGSM()(fmodel, images, labels, epsilons=epsilon)
        _, advs_pgd, success_pgd = LinfPGD(steps=n_steps_pgd, rel_stepsize=1/4)(fmodel, images, labels, epsilons=epsilon)
        if return_adjusted_fgsm:
            adjusted_fgsm = (((advs_fgsm - images) / torch.linalg.norm((advs_fgsm - images).reshape(advs_fgsm.shape[0], -1), dim=1).reshape(advs_fgsm.shape[0], 1, 1, 1))
            * torch.linalg.norm((advs_pgd - images).reshape(advs_pgd.shape[0], -1), dim=1).reshape(advs_fgsm.shape[0], 1, 1, 1)) + images
            _, _, success_adjusted_fgsm = FGSM()(fmodel, adjusted_fgsm, labels, epsilons=0) # this is a hack to get the successes. Can be done more efficiently
            successes_adjusted_fgsm.append(success_adjusted_fgsm)
        advs_fgsm = advs_fgsm.reshape(advs_fgsm.shape[0], -1)
        advs_pgd = advs_pgd.reshape(advs_pgd.shape[0], -1)
        cos = nn.CosineSimilarity(dim=1, eps=1e-12)
        cos_dif.append(cos(advs_fgsm, advs_pgd))
        dist = torch.linalg.norm(advs_fgsm - advs_pgd, dim=1, ord=2)
        distance.append(dist)
        successes_fgsm.append(success_fgsm)
        successes_pgd.append(success_pgd)
        
    if return_adjusted_fgsm:
        return torch.cat(cos_dif), torch.cat(distance), torch.cat(successes_fgsm), torch.cat(successes_pgd), torch.cat(successes_adjusted_fgsm)
    return torch.cat(cos_dif), torch.cat(distance), torch.cat(successes_fgsm), torch.cat(successes_pgd)

'''
Method that preforms an fgsm attack at a range of epsilons
'''
def multi_scale_fgsm(fmodel, images, labels, epsilon=0.03):
    scales = [epsilon * i/100 for i in range(1, 101)]    
    _, advs_fgsm, success_fgsm = FGSM()(fmodel, images, labels, epsilons=scales)
    return success_fgsm