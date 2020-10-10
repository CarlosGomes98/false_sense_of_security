# some code from RAI
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def fgsm_(model, x, target, eps, targeted=True, device='cpu', clip_min=None, clip_max=None):
    """Internal process for all FGSM and PGD attacks."""    
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_().to(device)
    # ... and make sure we are differentiating toward that variable
    input_.requires_grad_()

    # run the model and obtain the loss
    
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

def pgd_(model, x, target, step, eps, iters=7, targeted=True, device='cpu', clip_min=None, clip_max=None):
    projection_min = (x - eps).clamp_(min=clip_min)
    projection_max = (x + eps).clamp_(max=clip_max)
    
    # generate a random point in the +-eps box around x
    offset = torch.rand_like(x)
    offset = (offset*2*eps - eps)
    x = x + offset
    for _ in range(iters):
        x = fgsm_(model, x, target, step, targeted=targeted, device=device, clip_min=None, clip_max=None)
        # project
        x = torch.where(x < projection_min, projection_min, x)
        x = torch.where(x > projection_max, projection_max, x)
        
    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)
    return x

def gradient_information(model, data, target, step=0.01, eps=0.5, iters=20, targeted=False, device='cpu', clip_min=None, clip_max=None):
    adv = pgd_(model, data, target, step, eps, iters=iters, targeted=targeted, device=device, clip_min=clip_min, clip_max=clip_max).to(device)
    # only keep those for which an adversarial example was found
    new_labels = model(adv).argmax(axis=-1)
    adv_examples_index = new_labels != target
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
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    grad_information[adv_examples_index] = cos(grad, diff_vector)
    return grad_information

def adversarial_accuracy(model, dataset_loader, attack, **kwargs):
    correct = 0
    for batch_idx, (data, target) in enumerate(dataset_loader):
        data, target = data.to(device), target.to(device)
        adv = attack(model, data, target, 0.1, 0.5, iters=7, targeted=False, device=device, clip_min=normalized_min, clip_max=normalized_max)
        output = model(adv)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        if (batch_idx % 100 == 0):
            print('{} / {}'.format(batch_idx * dataset_loader.batch_size, len(dataset_loader.dataset)))
    print ((correct/len(dataset_loader.dataset) * 100))
    
if __name__ == "__main__":
    pass