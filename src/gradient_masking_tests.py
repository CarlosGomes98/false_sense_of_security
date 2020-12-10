import argparse
import math
from tqdm.notebook import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, FGSM, LinfDeepFoolAttack
from advertorch.attacks import LinfSPSAAttack
from robustbench.model_zoo.models import Carmon2019UnlabeledNet
from src.utils import adversarial_accuracy, fgsm_, random_step_
import eagerpy as ep
from src.Nets import CIFAR_Wide_Res_Net, CIFAR_Res_Net, CIFAR_Net


def run_masking_benchmarks(model,
                           test_dataset,
                           epsilon=0.06,
                           device="cpu",
                           batch_size=32,
                           epsilon_step=20):
    """
    This method runs through a checklist of potential indicators of gradient masking, as exposed in 
    "Obfuscated Gradients Give a False Sense of Security:
    Circumventing Defenses to Adversarial Examples"
    https://arxiv.org/pdf/1802.00420.pdf
    """
    epsilons = [epsilon * i / 100 for i in range(10, 200, epsilon_step)]
    pbar = tqdm(total=6, desc="Description")

    pbar.set_description("Computing Accuracy")
    acc = get_accuracy(model,
                       test_dataset,
                       epsilon=epsilon,
                       device=device,
                       batch_size=batch_size) * 100
    pbar.update(1)

    pbar.set_description("Computing FGSM Accuracy")
    fgsm_acc = np.array([
        get_accuracy(model,
                     test_dataset,
                     epsilon=ep,
                     device=device,
                     batch_size=batch_size,
                     attack=FGSM()) * 100 for ep in epsilons
    ])
    pbar.update(1)

    pbar.set_description("Computing PGD Accuracy")
    pgd_acc = get_accuracy(model,
                           test_dataset,
                           epsilon=epsilon,
                           device=device,
                           batch_size=batch_size,
                           attack=LinfPGD(steps=7, rel_stepsize=1 / 4)) * 100
    pgd_acc_small = get_accuracy(model,
                                 test_dataset,
                                 epsilon=epsilon / 2,
                                 device=device,
                                 batch_size=batch_size,
                                 attack=LinfPGD(steps=7,
                                                rel_stepsize=1 / 4)) * 100
    pgd_unbounded = get_accuracy(model,
                                 test_dataset,
                                 epsilon=1,
                                 device=device,
                                 batch_size=batch_size,
                                 attack=LinfPGD(steps=7,
                                                rel_stepsize=1 / 4)) * 100
    pbar.update(1)

    pbar.set_description("Computing SPSA Accuracy")
    spsa_acc = spsa_accuracy(model,
                             test_dataset,
                             eps=epsilon,
                             iters=10,
                             nb_sample=128,
                             batch_size=8,
                             device=device) * 100
    spsa_acc_small = spsa_accuracy(model,
                                   test_dataset,
                                   eps=epsilon / 2,
                                   iters=10,
                                   nb_sample=128,
                                   batch_size=8,
                                   device=device) * 100
    pbar.update(1)

    pbar.set_description("Computing Random Attack Accuracy")
    random_acc = np.array([
        get_random_accuracy(model,
                            test_dataset,
                            epsilon=ep,
                            device=device,
                            batch_size=batch_size) * 100 for ep in epsilons
    ])
    pbar.update(1)
    print("Model accuracy: {}%".format(acc))

    pbar.set_description("Plotting")
    print("PGD accuracy - eps = {}: {}%".format(epsilon, pgd_acc))
    print("PGD accuracy - eps = {}: {}%".format(epsilon / 2, pgd_acc))
    print("Unbounded PGD model accuracy: {}%".format(pgd_unbounded))

    print("SPSA accuracy - eps = {}: {}%".format(epsilon, spsa_acc))
    print("SPSA accuracy - eps = {}: {}%".format(epsilon / 2, spsa_acc_small))

    fig = plt.figure(figsize=(12, 8))
    plt.plot(epsilons, fgsm_acc, label='FGSM Accuracy')
    plt.plot(epsilons, random_acc, label='Random Attack Accuracy')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    pbar.update(1)
    pbar.close()


def get_accuracy(model,
                 test_dataset,
                 attack=None,
                 epsilon=0.03,
                 subset_size=10000,
                 device="cpu",
                 batch_size=32):
    """
    Reports the accuracy of the model, potentially under some attack (e.g. FGSM, PGD, ...)
    """
    fmodel = PyTorchModel(model, bounds=(0, 1))
    correct = 0
    subset = torch.utils.data.Subset(
        test_dataset,
        np.random.randint(0, len(test_dataset), size=subset_size).tolist())
    subset_loader = torch.utils.data.DataLoader(subset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=2)
    for images, labels in subset_loader:
        images, labels = images.to(device), labels.type(torch.cuda.LongTensor)
        if attack is None:
            correct += accuracy(fmodel, images, labels) * images.shape[0]
        else:
            _, _, success = attack(fmodel, images, labels, epsilons=epsilon)
            correct += (~success).sum().item()
    return correct / subset_size


def get_random_accuracy(model,
                        test_dataset,
                        epsilon=0.03,
                        device="cpu",
                        batch_size=128,
                        subset_size=10000):
    '''
    Calculate the accuracy of the model when subjected to a random attack.
    '''
    correct = 0
    subset = torch.utils.data.Subset(
        test_dataset,
        np.random.randint(0, len(test_dataset), size=subset_size).tolist())
    subset_loader = torch.utils.data.DataLoader(subset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=2)
    for images, labels in subset_loader:
        images, labels = images.to(device), labels.type(torch.cuda.LongTensor)
        adv = random_step_(model,
                           images,
                           eps=epsilon,
                           device=device,
                           clip_min=0,
                           clip_max=1)
        preds = model(adv).argmax(-1)
        correct += (preds == labels).sum().item()
    return correct / len(subset_loader.dataset)


def spsa_accuracy(model,
                  test_dataset,
                  eps=0.03,
                  iters=1,
                  nb_sample=128,
                  batch_size=8,
                  device="cpu",
                  subset_size=100):
    """
    Reports the accuracy of the model under the SPSA attack. This method is quite expensive, so a small subset_size is reccomended,
    particularly for deeper networks.
    """
    attack = LinfSPSAAttack(model,
                            eps,
                            nb_iter=iters,
                            nb_sample=nb_sample,
                            loss_fn=nn.CrossEntropyLoss(reduction='none'))
    subset = torch.utils.data.Subset(
        test_dataset,
        np.random.randint(0, len(test_dataset), size=subset_size).tolist())
    subset_loader = torch.utils.data.DataLoader(subset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=2)
    correct = 0
    for images, labels in subset_loader:
        images, labels = images.to(device), labels.type(torch.cuda.LongTensor)
        adv = attack.perturb(images, labels)
        preds = model(adv).argmax(-1)
        correct += (preds == labels).sum().item()
    return correct / len(subset_loader.dataset)


def gradient_norm(model, data_loader, device='cpu', subset_size=10000):
    """
    Computes the gradient norm w.r.t. the loss at the given points.
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


def linearization_error(model,
                        dataset,
                        subset=100,
                        batch_size=128,
                        n_perturbations=128 * 10,
                        epsilons=[0.03],
                        device='cpu'):
    """
    Estimates the 'linearizability' of a model by computing the linearization error over a series of randomly sampled points
    at set l-inf distances
    The idea is that attacks such as FGSM rely on linearizing the loss, which in turn relies on having a linearizable model
    if that linearizability is broken, attacks will have a harder time, while not necessarily ensuring a robust model
    """
    epsilon_errors = []
    for epsilon in epsilons:
        mean_errors = []
        for counter, data in enumerate(dataset):
            if counter > subset:
                break

            model.zero_grad()
            x = data[0].reshape((1, ) + data[0].shape).to(device)
            x.requires_grad_()
            y = model(x)[0, data[1]]
            g = torch.autograd.grad(y, x)[0]
            errors = []
            with torch.no_grad():
                for _ in range(math.ceil(n_perturbations / batch_size)):
                    perturbation = (torch.rand(
                        (batch_size, 3, 32, 32)) > 0.5).float().to(device)
                    perturbation[perturbation == 0] = -1
                    perturbation *= epsilon
                    #                     perturbation = torch.rand((batch_size, 3, 32, 32)).to(device)
                    #                     perturbation = perturbation * epsilon * 2
                    #                     perturbation = perturbation - epsilon
                    y_prime = model(x.repeat(batch_size, 1, 1, 1) +
                              perturbation)[:, data[1]]
                    approx = y.repeat(batch_size) + torch.sum(perturbation * g)
                    errors.append(torch.abs(y_prime - approx) / y_prime)
            mean_errors.append(torch.cat(errors).mean())

        epsilon_errors.append(torch.stack(mean_errors).mean())

    for epsilon, error in zip(epsilons, epsilon_errors):
        print("Epsilon {}: {} error".format(epsilon, error))


def gradient_information(model,
                         dataset,
                         iters=50,
                         device='cpu',
                         subset_size=1000):
    """
    Computes the cosine information between the gradient of point at the decision boundary w.r.t. the different in logits and the vector (point at decision boundary - original input point).

    For non gradient masked models, this point should be the closest one to the input that is at the decision boundary.
    Thus, we would expect these vectors to be +- collinear.
    """
    fmodel = PyTorchModel(model, bounds=(0, 1))
    attack = LinfDeepFoolAttack(overshoot=0.002, steps=iters)
    subset = torch.utils.data.Subset(
        dataset,
        np.random.randint(0, len(dataset), size=subset_size).tolist())
    subset_loader = torch.utils.data.DataLoader(subset,
                                                batch_size=128,
                                                shuffle=False,
                                                num_workers=2)
    grad_information_full = []
    for data, target in subset_loader:
        data = data.to(device)
        target = target.to(device)
        _, adv, success = attack(fmodel, data, target, epsilons=8)
        # only keep those for which an adversarial example was found
        new_labels = model(adv).argmax(axis=-1)
        adv_examples_index = new_labels != target
        # print("{} adv. examples found from {} data points".format(adv_examples_index.sum().item(), data.shape[0]))
        if adv_examples_index.sum() == 0:
            return None

        grad_information = torch.Tensor(adv.shape[0]).to(device)
        grad_information[~adv_examples_index] = float('nan')

        adv = adv[adv_examples_index].detach().clone()
        adv.requires_grad = True
        model.zero_grad()
        logits = model(adv)
        loss = torch.sum(logits[:, new_labels] - logits[:, target])
        loss.backward()
        grad = adv.grad.reshape(adv.shape[0], -1)
        diff_vector = (adv - data[adv_examples_index]).reshape(
            adv.shape[0], -1)
        cos = nn.CosineSimilarity(dim=1, eps=1e-18)
        grad_information[adv_examples_index] = cos(grad, diff_vector)
        grad_information_full.append(grad_information)
    return torch.cat(grad_information_full).mean()


def fgsm_pgd_cos_dif(model,
                     test_dataset,
                     epsilons=[0.03],
                     subset_size=1000,
                     device="cpu",
                     batch_size=32,
                     n_steps_pgd=7,
                     return_adjusted_fgsm=True):
    '''
    Method that evaluates how informative the gradients of the network are. Preforms pgd and fgsm and compares the solutions.
    Returns the cosine difference and euclidian distance between the solutions.
    Furthermore, the method computes and returns the success of the adjusted fgsm attack. It takes the output of the fgsm attack
    and rescales it to have the same norm as the pgd solution. This was implemented as it was noticed that the cosine similarity
    was often very close to 1, yet the norm was quite different.
    '''
    fmodel = PyTorchModel(model, bounds=(0, 1))
    for epsilon in epsilons:
        cos_dif = []
        distance = []
        successes_fgsm = []
        successes_pgd = []
        if return_adjusted_fgsm:
            successes_adjusted_fgsm = []
        subset = torch.utils.data.Subset(
            test_dataset,
            np.random.randint(0, len(test_dataset), size=subset_size).tolist())
        subset_loader = torch.utils.data.DataLoader(subset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=2)
        for images, labels in tqdm(subset_loader):
            images, labels = images.to(device), labels.type(
                torch.cuda.LongTensor)
            _, advs_fgsm, success_fgsm = FGSM()(fmodel,
                                                images,
                                                labels,
                                                epsilons=epsilon)
            _, advs_pgd, success_pgd = LinfPGD(steps=n_steps_pgd,
                                               rel_stepsize=1 / 4)(
                                                   fmodel,
                                                   images,
                                                   labels,
                                                   epsilons=epsilon)
            fgsm_perturbation = advs_fgsm - images
            pgd_perturbation = advs_pgd - images
            if return_adjusted_fgsm:
                adjusted_fgsm = ((fgsm_perturbation / torch.linalg.norm(
                    fgsm_perturbation.reshape(advs_fgsm.shape[0], -1), dim=1
                ).reshape(advs_fgsm.shape[0], 1, 1, 1)) * torch.linalg.norm(
                    pgd_perturbation.reshape(advs_pgd.shape[0], -1),
                    dim=1).reshape(advs_fgsm.shape[0], 1, 1, 1)) + images
                _, _, success_adjusted_fgsm = FGSM()(
                    fmodel, adjusted_fgsm, labels, epsilons=0
                )  # this is a hack to get the successes. Can be done more efficiently
                successes_adjusted_fgsm.append(success_adjusted_fgsm)
            fgsm_perturbation = fgsm_perturbation.reshape(
                fgsm_perturbation.shape[0], -1)
            pgd_perturbation = pgd_perturbation.reshape(
                pgd_perturbation.shape[0], -1)
            cos = nn.CosineSimilarity(dim=1, eps=1e-18)
            cos_dif.append(cos(fgsm_perturbation, pgd_perturbation))
            dist = torch.linalg.norm(fgsm_perturbation - pgd_perturbation,
                                     dim=1,
                                     ord=2)
            distance.append(dist)
            successes_fgsm.append(success_fgsm)
            successes_pgd.append(success_pgd)

        print("Epsilon = {}:".format(epsilon))
        cos_dif, distance, successes_fgsm, successes_pgd = torch.cat(
            cos_dif), torch.cat(distance), torch.cat(
                successes_fgsm), torch.cat(successes_pgd)
        print(
            "Mean Cosine Difference: {}, Mean Cosine Difference when FGSM does not succeed but PGD does: {}, Mean l2 Distance: {}"
            .format(cos_dif[successes_fgsm].mean(),
                    cos_dif[(~successes_fgsm & successes_pgd)].mean(),
                    dist.mean()))
        if return_adjusted_fgsm:
            successes_adjusted_fgsm = torch.cat(successes_adjusted_fgsm)
            print(
                "FGSM success: {}, PGD Success: {}, Rescaled FGSM success: {}".
                format(successes_fgsm.sum() / subset_size,
                       successes_pgd.sum() / subset_size,
                       successes_adjusted_fgsm.sum() / subset_size))
        else:
            print(
                "FGSM success: {}, PGD Success: {}".
                format(successes_fgsm.sum() / subset_size,
                       successes_pgd.sum() / subset_size))


def multi_scale_fgsm(fmodel, images, labels, epsilon=0.03):
    '''
    Method that preforms an fgsm attack at a range of epsilons
    '''
    scales = [epsilon * i / 100 for i in range(1, 101)]
    _, advs_fgsm, success_fgsm = FGSM()(fmodel,
                                        images,
                                        labels,
                                        epsilons=scales)
    return success_fgsm