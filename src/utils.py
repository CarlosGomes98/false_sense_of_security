# some of the code from RAI
# This file contains utility functions used throughout the project.
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, FGSM, LinfDeepFoolAttack


def fgsm_(model,
          x,
          target,
          eps=0.5,
          targeted=True,
          device='cpu',
          clip_min=None,
          clip_max=None,
          **kwargs):
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
    #perfrom either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad.sign()
    else:
        out = input_ + eps * input_.grad.sign()

    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


def step_ll_(model,
             x,
             target,
             eps=0.5,
             device='cpu',
             clip_min=None,
             clip_max=None,
             **kwargs):
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
    #perfrom either targeted or untargeted attack
    out = input_ - eps * input_.grad.sign()

    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


def random_step_(model,
                 x,
                 eps=0.5,
                 device='cpu',
                 clip_min=None,
                 clip_max=None,
                 **kwargs):
    """
    Returns examples generated from a step in a random direction.
    """
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_().to(device)
    rand_step = (torch.rand(x.shape) > 0.5).float().to(device)
    rand_step[rand_step == 0] = -1
    out = input_ + eps * rand_step

    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


def pgd_(model,
         x,
         target,
         step,
         eps,
         iters=7,
         targeted=True,
         device='cpu',
         clip_min=None,
         clip_max=None,
         random_step=True,
         early_stop=False):
    """
    Internal pgd attack used during training.

    Applies iterated steps of the fgsm attack, projecting back to the relevant domain after each step.
    """
    projection_min = x - eps
    projection_max = x + eps

    # generate a random point in the +-eps box around x
    if random_step:
        offset = torch.rand_like(x)
        offset = (offset * 2 * eps - eps)
        x = x + offset
    done = torch.BoolTensor(x.shape[0]).to(device)
    done[:] = False
    for i in range(iters):
        if early_stop:
            x_to_change = x.clone()[~done]
            x_to_change = fgsm_(model,
                                x_to_change,
                                target[~done],
                                eps=step,
                                targeted=targeted,
                                device=device,
                                clip_min=None,
                                clip_max=None)
            x[~done] = x_to_change
        else:
            x = fgsm_(model,
                      x,
                      target,
                      eps=step,
                      targeted=targeted,
                      device=device,
                      clip_min=None,
                      clip_max=None)
        # project
        x = torch.where(x < projection_min, projection_min, x)
        x = torch.where(x > projection_max, projection_max, x)
        if (clip_min is not None) or (clip_max is not None):
            x.clamp_(min=clip_min, max=clip_max)

        # check for adv examples
        if early_stop:
            with torch.no_grad():
                new_labels = model(x).argmax(axis=-1)
                done = new_labels != target
            if done.all():
                break
    return x


def adversarial_accuracy(model,
                         dataset_loader,
                         attack=pgd_,
                         iters=20,
                         eps=0.5,
                         step=0.1,
                         random_step=False,
                         device="cpu"):
    """
    Compute the adversarial accuracy of a model.

    Deprecated, moved to using foolbox, advertorch.
    """
    correct = 0
    for batch_idx, (data, target) in enumerate(dataset_loader):
        data, target = data.to(device), target.to(device)
        adv = attack(model,
                     data,
                     target,
                     step=step,
                     eps=eps,
                     iters=iters,
                     targeted=False,
                     device=device,
                     clip_min=0,
                     clip_max=1,
                     random_step=random_step)
        output = model(adv)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        if (batch_idx % 10 == 0):
            print('{} / {}'.format(batch_idx * dataset_loader.batch_size,
                                   len(dataset_loader.dataset)))
    return (correct / len(dataset_loader.dataset) * 100)


if __name__ == "__main__":
    pass