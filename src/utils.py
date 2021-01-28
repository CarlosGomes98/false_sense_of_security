# some of the code from RAI
# This file contains utility functions used throughout the project.
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, FGSM, LinfDeepFoolAttack


def fgsm_(
    model,
    x,
    target,
    eps=0.5,
    targeted=True,
    device="cpu",
    clip_min=None,
    clip_max=None,
    **kwargs
):
    """
    Internal process for all FGSM and PGD attacks used during training.
    
    Returns the adversarial examples crafted from the inputs using the FGSM attack.
    """
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_().to(device)
    # ... and make sure we are differentiating toward that variable
    input_.requires_grad_()

    # run the model and obtain the los
    model.zero_grad()
    logits = model(input_)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    # perfrom either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad.sign()
    else:
        out = input_ + eps * input_.grad.sign()

    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


def step_ll_(
    model, x, target, eps=0.5, device="cpu", clip_min=None, clip_max=None, **kwargs
):
    """
    Implementation of the Step_LL attack, minimizing the loss for the least likely class
    """
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_().to(device)
    # ... and make sure we are differentiating toward that variable
    input_.requires_grad_()

    # run the model and obtain the los
    model.zero_grad()
    logits = model(input_)
    least_likely = torch.argmin(logits, dim=1)
    loss = nn.CrossEntropyLoss()(logits, least_likely)
    loss.backward()
    # perfrom either targeted or untargeted attack
    out = input_ - eps * input_.grad.sign()

    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


def random_step_(
    model, x, eps=0.5, device="cpu", clip_min=None, clip_max=None, **kwargs
):
    """
    Returns examples generated from a step in a random direction.
    """
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_().to(device)
    rand_step = (torch.rand(x.shape) > 0.5).float().to(device)
    rand_step[rand_step == 0] = -1
    out = input_ + eps * rand_step

    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


def pgd_(
    model,
    x,
    target,
    eps=0.03,
    step=1/4,
    iters=7,
    targeted=True,
    device="cpu",
    clip_min=None,
    clip_max=None,
    random_step=True,
    report_steps=False,
    project=True
):
    """
    Internal pgd attack used during training.

    Applies iterated steps of the fgsm attack, projecting back to the relevant domain after each step.
    """
    projection_min = x - eps
    projection_max = x + eps
    if report_steps:
        steps = []
        # arrived = torch.ones()
    # generate a random point in the +-eps box around x
    if random_step:
        offset = torch.rand_like(x)
        offset = offset * 2 * eps - eps
        x = x + offset

    for i in range(iters):
        new_x = fgsm_(
            model,
            x,
            target,
            eps=eps * step,
            targeted=targeted,
            device=device,
            clip_min=None,
            clip_max=None,
        )
        # project
        if project:
            new_x = torch.where(new_x < projection_min, projection_min, new_x)
            new_x = torch.where(new_x > projection_max, projection_max, new_x)
        if (clip_min is not None) or (clip_max is not None):
            new_x.clamp_(min=clip_min, max=clip_max)
        model.zero_grad()
        steps.append(new_x - x)
        x = new_x
    if not report_steps:
        return x
    else:
        return x, steps


def adversarial_accuracy(
    model,
    dataset_loader,
    attack=pgd_,
    iters=20,
    eps=0.5,
    step=1 / 8,
    random_step=False,
    device="cpu",
):
    """
    Compute the adversarial accuracy of a model.

    Deprecated, moved to using foolbox, advertorch.
    """
    correct = 0
    for batch_idx, (data, target) in enumerate(dataset_loader):
        data, target = data.to(device), target.to(device)
        adv = attack(
            model,
            data,
            target,
            eps=eps,
            step=step,
            iters=iters,
            targeted=False,
            device=device,
            clip_min=0,
            clip_max=1,
            random_step=random_step,
        )
        output = model(adv)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % 10 == 0:
            print(
                "{} / {}".format(
                    batch_idx * dataset_loader.batch_size, len(dataset_loader.dataset)
                )
            )
    return correct / len(dataset_loader.dataset) * 100


def plot_along_grad(
    perturbations, model, datapoint, target, batch_size, device="cpu", axis=None, draw=True
):

    """
    Method that measures the loss when following the gradient direction from a data point at increasing epsilons.
    Also visualizes it if draw is True.
    """
    losses = []
    datapoint = datapoint.unsqueeze(0)
    datapoint = datapoint.to(device)
    datapoint.requires_grad_()
    target = torch.LongTensor([target]).to(device)
    output = model(datapoint)
    predicted_class = output.argmax(-1).type(torch.LongTensor).to(device)
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    loss = ce(output, predicted_class)
    direction = torch.autograd.grad(loss, datapoint, only_inputs=True)[0].sign()
    diff_class = []
    with torch.no_grad():
        for batch in range(0, perturbations.shape[0], batch_size):
            cur_batch_size = min(batch_size, perturbations.shape[0] - batch)
            cur_target = predicted_class.repeat(cur_batch_size)
            data = datapoint.repeat(
                cur_batch_size, 1, 1, 1
            ) + direction * perturbations[batch : batch + cur_batch_size].reshape(
                -1, 1, 1, 1
            )
            output = model(data)
            diff_class.append(output.argmax(-1) != predicted_class)
            loss = ce(output, cur_target)
            losses.append(loss)
    losses = torch.cat(losses).detach().cpu()
    diff_class = torch.cat(diff_class).detach().cpu()
    first_diff_class = find_first_true(diff_class)

    if draw:
        if axis is None:
            plt.ylim(0, 30)
            plt.plot(perturbations.detach().cpu().numpy(), losses, alpha=0.05, color="blue")
            plt.xlabel("Epsilon")
            plt.ylabel("Loss")
        else:
            axis.set_ylim(0, 30)
            axis.plot(perturbations.detach().cpu().numpy(), losses, alpha=0.05, color="blue")
            axis.set_xlabel("Epsilon")
            axis.set_ylabel("Loss")
    return losses, first_diff_class

def find_first_true(vector):
    for i in range(vector.shape[0]):
        if vector[i].item():
            return i
    
    return -1

def plot_along_grad_n(model, datasets, batch_size, n, device="cpu", draw=100):
    """
    Makes use of plot_along_grad for several data points
    """
    perturbations = torch.arange(0, 0.16, 0.002).to(device)
    fig, ax = plt.subplots(1, len(datasets), figsize=(12, 5))
    for axis, dataset in zip(ax, datasets):
        datapoint_indexes = torch.randint(0, len(dataset), (n,))
        losses_total = []
        boundaries_total = []
        no_boundary_crossed = 0
        for count, index in enumerate(datapoint_indexes):
            loss, boundary = plot_along_grad(
                perturbations,
                model,
                dataset[index][0],
                dataset[index][1],
                batch_size,
                axis=axis,
                device=device,
                draw=count<draw
            )
            losses_total.append(loss)
            if boundary == -1:
                no_boundary_crossed += 1
            else:
                boundaries_total.append(perturbations[boundary].detach().cpu().item())

        boundaries_total = np.array(boundaries_total)
        losses_total = torch.stack(losses_total)
        losses_mean = losses_total.mean(axis=0).detach().cpu().numpy()
        losses_std = losses_total.std(axis=0).detach().cpu().numpy()
        axis.set_ylim(0, 30)
        axis.plot(perturbations.detach().cpu().numpy(), losses_mean, alpha=1, color="red")
        axis.fill_between(
            perturbations.detach().cpu().numpy(),
            losses_mean - losses_std,
            losses_mean + losses_std,
            color="red",
            alpha=0.3,
        )
        axis.set_xlabel("Epsilon")
        axis.set_ylabel("Loss")

        boundaries_mean = boundaries_total.mean()
        boundaries_std = boundaries_total.std()
        axis.axvline(boundaries_mean, alpha=1, color='green')
        axis.axvline(boundaries_mean + boundaries_std, alpha=0.5, color='green')
        axis.axvline(boundaries_mean - boundaries_std, alpha=0.5, color='green')
        print(no_boundary_crossed)
    plt.show()


def compare_models_on_measure(
    measure_function, models, labels, data_loader, device="cpu", height=2, bins=100, lim=None, **kwargs
):
    """
    Utility function that compares a metric over several models. Displays a histogram for each model, in a grid.
    """
    measures = [
        measure_function(model, data_loader, device=device, **kwargs).detach().cpu().numpy()
        for model in models
    ]
    width = math.ceil(len(models) / height)
    fig, ax = plt.subplots(height, width, figsize=(15, 15))
    for index, measure in enumerate(measures):
        axis = ax[index // width, index % width]
        axis.hist(measure, bins=bins)
        axis.set_title(labels[index])
        if lim is not None:
            axis.set_xlim(lim[0], lim[1])
        axis.text(
            0.5,
            -0.13,
            "Max: {:.3f}, Mean: {:.3f}, Median: {:.3f}".format(
                measure.max().item(),
                measure.mean().item(),
                np.median(measure).item(),
            ),
            size=12,
            ha="center",
            transform=axis.transAxes,
        )
    plt.show()


if __name__ == "__main__":
    pass
