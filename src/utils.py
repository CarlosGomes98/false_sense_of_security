# some of the code from RAI
# This file contains utility functions used throughout the project.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def fgsm_(model, x, target, eps=0.5, targeted=True, device='cpu', clip_min=None, clip_max=None, **kwargs):
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

def pgd_(model, x, target, step, eps, iters=7, targeted=True, device='cpu', clip_min=None, clip_max=None, random_step=True, early_stop=False):
    """
    Internal pgd attack used during training.

    Applies iterated steps of the fgsm attack, projecting back to the relevant domain after each step.
    """
    projection_min = x - eps
    projection_max = x + eps
    
    # generate a random point in the +-eps box around x
    if random_step:
        offset = torch.rand_like(x)
        offset = (offset*2*eps - eps)
        x = x + offset
    done = torch.BoolTensor(x.shape[0]).to(device)
    done[:] = False
    for i in range(iters):
        if early_stop:
            x_to_change = x.clone()[~done]
            x_to_change = fgsm_(model, x_to_change, target[~done], eps=step, targeted=targeted, device=device, clip_min=None, clip_max=None)
            x[~done] = x_to_change
        else:
            x = fgsm_(model, x, target, eps=step, targeted=targeted, device=device, clip_min=None, clip_max=None)
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

def gradient_information(model, data, target, step=0.01, eps=0.5, iters=20, targeted=False, device='cpu', clip_min=None, clip_max=None):
    """
    Computes the cosine information between the gradient of point at the decision boundary w.r.t. the loss and the vector (point at decision boundary - original input point).

    For non gradient masked models, this point should be the closest one to the input that is at the decision boundary.
    Thus, we would expect these vectors to be +- collinear.

    TODO: Incorrect implementation!!!!!! Do not use PGD, use something like deepfool. And do not use loss function for decision boundary, but rather difference in logits.
    TODO: Move to metrics
    """
    
    adv = pgd_(model, data, target, step, eps, iters=iters, targeted=targeted, device=device, clip_min=clip_min, clip_max=clip_max, random_step=False, early_stop=True).to(device)
    # only keep those for which an adversarial example was found
    new_labels = model(adv).argmax(axis=-1)
    adv_examples_index = new_labels != target
    print("{} adv. examples found from {} data points".format(adv_examples_index.sum().item(), data.shape[0]))
    if adv_examples_index.sum() == 0:
        return None
    
    grad_information = torch.Tensor(adv.shape[0]).to(device)
    grad_information[~adv_examples_index] = float('nan')
    
    adv = adv[adv_examples_index].detach().clone()
    adv.requires_grad = True

    model.zero_grad()
    logits = model(adv)
    loss = nn.CrossEntropyLoss()(logits, target[adv_examples_index])
    loss.backward()

    grad = adv.grad.reshape(adv.shape[0], -1)
    diff_vector = (adv - data[adv_examples_index]).reshape(adv.shape[0], -1)
    cos = nn.CosineSimilarity(dim=1, eps=1e-12)
    grad_information[adv_examples_index] = cos(grad, diff_vector)
    return grad_information

def adversarial_accuracy(model, dataset_loader, attack=pgd_, iters=20, eps=0.5, step=0.1, random_step=False, device="cpu"):
    """
    Compute the adversarial accuracy of a model.

    Deprecated, moved to using foolbox, advertorch.
    """
    correct = 0
    for batch_idx, (data, target) in enumerate(dataset_loader):
        data, target = data.to(device), target.to(device)
        adv = attack(model, data, target, step=step, eps=eps, iters=iters, targeted=False, device=device, clip_min=0, clip_max=1, random_step=random_step)
        output = model(adv)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        if (batch_idx % 10 == 0):
            print('{} / {}'.format(batch_idx * dataset_loader.batch_size, len(dataset_loader.dataset)))
    return (correct/len(dataset_loader.dataset) * 100)
    
if __name__ == "__main__":
    pass