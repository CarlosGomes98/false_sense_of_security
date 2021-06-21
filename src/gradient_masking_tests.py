import argparse
import math
from tqdm import tqdm
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
from src.utils import adversarial_accuracy, fgsm_, random_step_, pgd_
from src.load_architecture import CIFAR_Wide_Res_Net, CIFAR_Res_Net, CIFAR_Net


def run_masking_benchmarks(
    model,
    test_dataset,
    epsilon=0.06,
    device="cpu",
    batch_size=128,
    return_dict=False,
    save_fig=None,
    subset_size=10000,
    report_results=False
):
    """
    This method runs through a checklist of potential indicators of gradient masking, as exposed in 
    "Obfuscated Gradients Give a False Sense of Security:
    Circumventing Defenses to Adversarial Examples"
    https://arxiv.org/pdf/1802.00420.pdf
    """

    results = {}
    epsilons = [epsilon * i / 100 for i in range(10, 200, 10)]
    results["Epsilons Range"] = np.array(epsilons)
    pbar = tqdm(total=6, desc="Description")

    pbar.set_description("Computing Accuracy")
    acc = (
        get_accuracy(
            model, test_dataset, attack=None, device=device, batch_size=batch_size, subset_size=subset_size
        )
        * 100
    )

    results["Clean Accuracy"] = acc
    pbar.update(1)

    pbar.set_description("Computing FGSM Accuracy")
    fgsm_acc = np.array(
        [
            get_accuracy(
                model,
                test_dataset,
                epsilon=ep,
                device=device,
                batch_size=batch_size,
                attack=FGSM(),
                subset_size=subset_size
            )
            * 100
            for ep in epsilons
        ]
    )
    pbar.update(1)
    results["FGSM Accuracy - Range"] = fgsm_acc
    results["FGSM Accuracy eps: 0.06"] = fgsm_acc[9]
    results["FGSM Accuracy eps: 0.03"] = fgsm_acc[4]

    pbar.set_description("Computing PGD Accuracy")
    pgd_acc = (
        get_accuracy(
            model,
            test_dataset,
            epsilon=epsilon,
            device=device,
            batch_size=batch_size,
            attack=LinfPGD(steps=7, rel_stepsize=1 / 4),
            subset_size=subset_size
        )
        * 100
    )
    results["PGD Accuracy eps: 0.06"] = pgd_acc

    pgd_acc_small = (
        get_accuracy(
            model,
            test_dataset,
            epsilon=epsilon / 2,
            device=device,
            batch_size=batch_size,
            attack=LinfPGD(steps=7, rel_stepsize=1 / 4),
            subset_size=subset_size
        )
        * 100
    )
    results["PGD Accuracy eps: 0.03"] = pgd_acc_small

    pgd_unbounded = (
        get_accuracy(
            model,
            test_dataset,
            epsilon=0.5,
            device=device,
            batch_size=batch_size,
            attack=LinfPGD(steps=7, rel_stepsize=1 / 4),
            subset_size=subset_size
        )
        * 100
    )
    pbar.update(1)
    results["PGD Accuracy eps: Unbounded"] = pgd_unbounded

    pbar.set_description("Computing SPSA Accuracy")
    spsa_acc = (
        spsa_accuracy(
            model,
            test_dataset,
            eps=epsilon,
            iters=15, #10
            nb_sample=256, #128
            batch_size=8,
            device=device,
            subset_size=500, #500
        )
        * 100
    )
    results["SPSA Accuracy eps: 0.06"] = spsa_acc

    spsa_acc_small = (
        spsa_accuracy(
            model,
            test_dataset,
            eps=epsilon / 2,
            iters=15,
            nb_sample=256,
            batch_size=8,
            device=device,
            subset_size=500, #500
        )
        * 100
    )
    results["SPSA Accuracy eps: 0.03"] = spsa_acc_small
    pbar.update(1)

    pbar.set_description("Computing Random Attack Accuracy")
    random_acc = np.array(
        [
            get_random_accuracy(
                model, test_dataset, epsilon=ep, device=device, batch_size=batch_size, subset_size=subset_size
            )
            * 100
            for ep in epsilons
        ]
    )
    results["Random Accuracy - Range"] = random_acc
    pbar.update(1)
    
    
    pbar.set_description("Plotting")
    if report_results:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(epsilons, fgsm_acc, label="FGSM Accuracy")
        ax.plot(epsilons, random_acc, label="Random Attack Accuracy")
        ax.set(xlabel="Epsilon", ylabel="Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        plt.show()
        print("Model accuracy: {}%".format(acc))
        print("FGSM accuracy - eps = {}: {}%".format(epsilon, fgsm_acc[9]))
        print("FGSM accuracy - eps = {}: {}%".format(epsilon / 2, fgsm_acc[4]))
        print("PGD accuracy - eps = {}: {}%".format(epsilon, pgd_acc))
        print("PGD accuracy - eps = {}: {}%".format(epsilon / 2, pgd_acc_small))
        print("Unbounded PGD model accuracy: {}%".format(pgd_unbounded))

        print("SPSA accuracy - eps = {}: {}%".format(epsilon, spsa_acc))
        print("SPSA accuracy - eps = {}: {}%".format(epsilon / 2, spsa_acc_small))
    
    if save_fig is not None:
        fig.savefig(save_fig, dpi=fig.dpi)
       
    pbar.update(1)
    pbar.close()

    if return_dict:
        return results


def get_accuracy(
    model,
    test_dataset,
    attack=None,
    epsilon=8/256,
    subset_size=10000,
    device="cpu",
    batch_size=128,
):
    """
    Reports the accuracy of the model, potentially under some attack (e.g. FGSM, PGD, ...)
    """

    fmodel = PyTorchModel(model, bounds=(0, 1), device=device)
    correct = 0
    subset = torch.utils.data.Subset(
        test_dataset, np.random.randint(0, len(test_dataset), size=subset_size).tolist()
    )
    subset_loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    for images, labels in subset_loader:
        images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
        if attack is None:
            correct += accuracy(fmodel, images, labels) * images.shape[0]
        else:
            _, _, success = attack(fmodel, images, labels, epsilons=epsilon)
            correct += (~success).sum().item()
    return correct / subset_size


def get_random_accuracy(
    model, test_dataset, epsilon=0.03, device="cpu", batch_size=128, subset_size=10000
):
    """
    Calculate the accuracy of the model when subjected to a random attack.
    """
    correct = 0
    subset = torch.utils.data.Subset(
        test_dataset, np.random.randint(0, len(test_dataset), size=subset_size).tolist()
    )
    subset_loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    for images, labels in subset_loader:
        images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
        adv = random_step_(
            model, images, eps=epsilon, device=device, clip_min=0, clip_max=1
        )
        preds = model(adv).argmax(-1)
        correct += (preds == labels).sum().item()
    return correct / len(subset_loader.dataset)


def spsa_accuracy(
    model,
    test_dataset,
    eps=0.03,
    iters=10,
    nb_sample=128,
    batch_size=8,
    device="cpu",
    subset_size=100,
):
    """
    Reports the accuracy of the model under the SPSA attack. This method is quite expensive, so a small subset_size is reccomended,
    particularly for deeper networks.
    """
    attack = LinfSPSAAttack(
        model,
        eps,
        nb_iter=iters,
        nb_sample=nb_sample,
        loss_fn=nn.CrossEntropyLoss(reduction="none"),
    )
    subset = torch.utils.data.Subset(
        test_dataset, np.random.randint(0, len(test_dataset), size=subset_size).tolist()
    )
    subset_loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    correct = 0
    for images, labels in subset_loader:
        images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
        adv = attack.perturb(images, labels)
        preds = model(adv).argmax(-1)
        correct += (preds == labels).sum().item()
    return correct / len(subset_loader.dataset)


def gradient_norm(model, dataset, device="cpu", subset_size=1000, return_dict=True, batch_size=128):
    """
    Computes the gradient norm w.r.t. the loss at the given points.
    """

    subset = torch.utils.data.Subset(
        dataset, np.random.randint(0, len(dataset), size=subset_size).tolist()
    )
    subset_loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    grad_norms = []
    for (data, target) in subset_loader:
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

    if return_dict:
        return {"Gradient Norm": grad_norm.detach().cpu().numpy()}
    return grad_norm

def jacobian_norm(model, dataset, device="cpu", subset_size=1000, return_dict=True, batch_size=128):
    """
    Computes the jacobian norm w.r.t. the loss at the given points.
    """

    subset = torch.utils.data.Subset(
        dataset, np.random.randint(0, len(dataset), size=subset_size).tolist()
    )
    subset_loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    jac_norms = []
    for (data, target) in subset_loader:
        data = data.to(device).requires_grad_()
        output = model(data)
        norms = torch.zeros(data.shape[0]).to(device)
        for i in range(10):
            model.zero_grad()
            logit = output[:, i]
            gradient = torch.autograd.grad(outputs=logit, inputs=data, grad_outputs=torch.ones_like(logit), only_inputs=True, create_graph=True)[0]
            gradient = gradient.view(data.shape[0], -1)
            norms += torch.linalg.norm(gradient, dim=1)**2
        jac_norms.append(norms.sqrt())
    jac_norm = torch.cat(jac_norms)
    if return_dict:
        return {"Gradient Norm": jac_norm.detach().cpu().numpy()}
    return jac_norm

def linearization_error(
    model,
    dataset,
    subset_size=500,
    batch_size=128,
    n_perturbations=128 * 2,
    epsilons=[0.03, 0.06],
    device="cpu",
    loss=False,
    return_dict=False,
):
    """
    Estimates the 'linearizability' of a model by computing the linearization error over a series of randomly sampled points
    at set l-inf distances
    The idea is that attacks such as FGSM rely on linearizing the loss, which in turn relies on having a linearizable model
    if that linearizability is broken, attacks will have a harder time, while not necessarily ensuring a robust model

    Specifically, we calculate the linearization error for the logit of the target class
    """
    epsilon_errors = {}
    datapoint_indexes = torch.randint(0, len(dataset), (subset_size,))
    ce = nn.CrossEntropyLoss(reduction="none")
    for epsilon in epsilons:
        mean_errors = []
        no_datapoints_skipped = 0
        for index in datapoint_indexes:
            perturbations_skipped = 0
            data = dataset[index]
            model.zero_grad()
            x = data[0].reshape((1,) + data[0].shape).to(device)
            x.requires_grad_()
            logits = model(x)
            target = torch.LongTensor([data[1]]).repeat(batch_size).to(device)
            if loss:
                y = ce(logits, torch.LongTensor([data[1]]).to(device))
            else:
                y = logits[0, data[1]]
            g = torch.autograd.grad(y, x)[0]
            errors = []
            with torch.no_grad():
                # if n_perturbations is not divisible by batch size, just do one more batch (overshoot)
                for _ in range(math.ceil(n_perturbations / batch_size)):
                    perturbation = (
                        (torch.rand((batch_size, 3, 32, 32)) > 0.5).float().to(device)
                    )

                    perturbation[perturbation == 0] = -1
                    perturbation *= epsilon
                    #                     perturbation = torch.rand((batch_size, 3, 32, 32)).to(device)
                    #                     perturbation = perturbation * epsilon * 2
                    #                     perturbation = perturbation - epsilon
                    logits_prime = model(x.repeat(batch_size, 1, 1, 1) + perturbation)
                    if loss:
                        y_prime = ce(logits_prime, target).reshape(-1, 1)
                    else:
                        y_prime = logits_prime[:, data[1]]

                    # since we are dividing by y_prime, it cannot be 0. Remove perturbations where that is the case
                    mask = y_prime != 0
                    perturbations_skipped += (~mask).sum().item()
                    y_prime = y_prime[mask]
                    approx = y.repeat(y_prime.shape[0]) + torch.sum(perturbation * g)
                    errors.append(torch.abs((approx - y_prime) / torch.abs(y_prime)))
            # if all the perturbations result in a y_prime of 0, skip the datapoint completely
            if perturbations_skipped == n_perturbations:
                no_datapoints_skipped += 1
                mean_errors.append(np.NaN)
            else:
                mean_errors.append(torch.cat(errors).mean().item())

        epsilon_errors["Linearization Error eps: " + str(epsilon)] = np.array(mean_errors)
        # print(no_datapoints_skipped)

    return epsilon_errors


def gradient_information(
    model,
    dataset,
    iters=5,
    device="cpu",
    subset_size=1000,
    batch_size=128,
    grad_collinearity=True,
    return_dict=False,
):
    """
    Computes the cosine information between the gradient of point at the decision boundary w.r.t. the different in logits and the vector (point at decision boundary - original input point).

    For non gradient masked models, this point should be the closest one to the input that is at the decision boundary.
    Thus, we would expect these vectors to be +- collinear.

    Another approach is to take this same gradient at the datapoint, and check collinearity with the gradient at the boundary.
    """
    fmodel = PyTorchModel(model, bounds=(0, 1))
    attack = LinfDeepFoolAttack(steps=iters)
    subset = torch.utils.data.Subset(
        dataset, np.random.randint(0, len(dataset), size=subset_size).tolist()
    )
    subset_loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    grad_information_full = []
    for data, target in subset_loader:
        data = data.to(device).requires_grad_()
        target = target.to(device)
        logits = model(data)
        predicted = logits.argmax(-1)
        # adv are points that should be close to the boundary
        _, adv, _ = attack(fmodel, data.detach(), predicted, epsilons=10)

        # only keep those for which an adversarial example was found: new label != originally predicted label
        with torch.no_grad():
            new_labels = model(adv).argmax(-1)

        adv_examples_index = new_labels != predicted
        # print("{} adv. examples found from {} data points".format(adv_examples_index.sum().item(), data.shape[0]))
        if adv_examples_index.sum() == 0:
            continue

        # remove examples that did not make it to the boundary (usually very few)
        adv = adv[adv_examples_index].detach().clone()
        new_labels = new_labels[adv_examples_index]
        predicted = predicted[adv_examples_index]

        # find the gradient at the boundary
        adv.requires_grad = True
        model.zero_grad()
        adv_logits = model(adv)
        loss = torch.sum(
            adv_logits.gather(1, new_labels.view(-1, 1))
            - adv_logits.gather(1, predicted.view(-1, 1))
        )
        loss.backward()
        grad = adv.grad.reshape(adv.shape[0], -1)

        # either take the same gradient at the datapoint or take the perturbation vector as the second term
        # for cosine similarity
        if grad_collinearity:
            model.zero_grad()
            filtered_logits = logits[adv_examples_index]
            loss = torch.sum(
                filtered_logits.gather(1, new_labels.view(-1, 1))
                - filtered_logits.gather(1, predicted.view(-1, 1))
            )
            loss.backward()
            vector = data.grad[adv_examples_index].reshape(grad.shape[0], -1)
        else:
            vector = (adv - data[adv_examples_index]).reshape(adv.shape[0], -1)
        cos = nn.CosineSimilarity(dim=1, eps=1e-18)
        grad_information = cos(grad, vector)
        grad_information_full.append(grad_information)
    if len(grad_information_full) == 0:
        return None
    if return_dict:
        return {
            "Gradient Information": torch.cat(grad_information_full)
            .detach()
            .cpu()
            .numpy()
        }
    return torch.cat(grad_information_full)


def fgsm_pgd_cos_dif(
    model,
    test_dataset,
    epsilons=[0.03, 0.06],
    subset_size=1000,
    device="cpu",
    batch_size=128,
    n_steps_pgd=10,
    return_adjusted_fgsm=False,
    return_dict=False,
):
    """
    Method that evaluates how informative the gradients of the network are. Preforms pgd and fgsm and compares the solutions.
    Returns the cosine difference and euclidian distance between the solutions.
    Furthermore, the method computes and returns the success of the adjusted fgsm attack. It takes the output of the fgsm attack
    and rescales it to have the same norm as the pgd solution. This was implemented as it was noticed that the cosine similarity
    was often very close to 1, yet the norm was quite different.
    """
    fmodel = PyTorchModel(model, bounds=(0, 1))
    results = {}
    for epsilon in epsilons:
        cos_dif = []
        distance = []
        successes_fgsm = []
        successes_pgd = []
        if return_adjusted_fgsm:
            successes_adjusted_fgsm = []
        subset = torch.utils.data.Subset(
            test_dataset,
            np.random.randint(0, len(test_dataset), size=subset_size).tolist(),
        )
        subset_loader = torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        for images, labels in subset_loader:
            images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
            _, advs_fgsm, success_fgsm = FGSM()(
                fmodel, images, labels, epsilons=epsilon
            )
            _, advs_pgd, success_pgd = LinfPGD(steps=n_steps_pgd, rel_stepsize=1 / 4)(
                fmodel, images, labels, epsilons=epsilon
            )
            fgsm_perturbation = advs_fgsm - images
            pgd_perturbation = advs_pgd - images
            if return_adjusted_fgsm:
                adjusted_fgsm = (
                    (
                        fgsm_perturbation
                        / torch.linalg.norm(
                            fgsm_perturbation.reshape(advs_fgsm.shape[0], -1), dim=1
                        ).reshape(advs_fgsm.shape[0], 1, 1, 1)
                    )
                    * torch.linalg.norm(
                        pgd_perturbation.reshape(advs_pgd.shape[0], -1), dim=1
                    ).reshape(advs_fgsm.shape[0], 1, 1, 1)
                ) + images
                _, _, success_adjusted_fgsm = FGSM()(
                    fmodel, adjusted_fgsm, labels, epsilons=0
                )  # this is a hack to get the successes. Can be done more efficiently
                successes_adjusted_fgsm.append(success_adjusted_fgsm)
            fgsm_perturbation = fgsm_perturbation.reshape(
                fgsm_perturbation.shape[0], -1
            )
            pgd_perturbation = pgd_perturbation.reshape(pgd_perturbation.shape[0], -1)
            cos = nn.CosineSimilarity(dim=1, eps=1e-18)
            cos_dif.append(cos(fgsm_perturbation, pgd_perturbation))
            dist = torch.linalg.norm(fgsm_perturbation - pgd_perturbation, dim=1, ord=2)
            distance.append(dist)
            successes_fgsm.append(success_fgsm)
            successes_pgd.append(success_pgd)
        successes_fgsm = torch.cat(successes_fgsm)
        successes_pgd = torch.cat(successes_pgd)
        results["FGSM PGD Cosine Similarity eps: " + str(epsilon)] = (
            torch.cat(cos_dif).detach().cpu().numpy()
        )
        if return_adjusted_fgsm:
            successes_adjusted_fgsm = torch.cat(successes_adjusted_fgsm)
        #     print(
        #         "Epsilon {} -- Cos Sim: {}, FGSM success: {}, PGD Success: {}, Rescaled FGSM success: {}".format(
        #             epsilon,
        #             results[str(epsilon)].mean().item(),
        #             successes_fgsm.sum() / subset_size,
        #             successes_pgd.sum() / subset_size,
        #             successes_adjusted_fgsm.sum() / subset_size,
        #         )
        #     )
        # else:
        #     print(
        #         "Epsilon {} -- Cos Sim: {}, FGSM success: {}, PGD Success: {}".format(
        #             epsilon,
        #             results[str(epsilon)].mean().item(),
        #             successes_fgsm.sum() / subset_size,
        #             successes_pgd.sum() / subset_size,
        #         )
        #     )
    return results


def multi_scale_fgsm(fmodel, images, labels, epsilon=0.03):
    """
    Method that preforms an fgsm attack at a range of epsilons
    """
    scales = [epsilon * i / 100 for i in range(1, 101)]
    _, advs_fgsm, success_fgsm = FGSM()(fmodel, images, labels, epsilons=scales)
    return success_fgsm


# should i take loss w.r.t. target or to currently predicted class? seyed suggested currently predicted i think. im not sure
def pgd_collinearity(
    model,
    dataset,
    epsilon=0.03,
    device="cpu",
    subset_size=1000,
    batch_size=128,
    random_step=False,
    sequential=False,
    return_dict=False
):
    """
    Compute a measure of the collinearity of pgd steps.
    Returns a torch tensor of size subset_size x number of pgd steps
    if sequential is true: computes the cosine similarity between subsequent steps
    if false, computes the cosine similarity between every step and the first step
    """
    subset = torch.utils.data.Subset(
        dataset, np.random.randint(0, len(dataset), size=subset_size).tolist()
    )

    subset_loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    cos = nn.CosineSimilarity(dim=1, eps=1e-18)
    result = []
    for images, labels in subset_loader:
        images = images.to(device)
        labels = labels.type(torch.LongTensor).to(device)

        with torch.no_grad():
            predicted = model(images).argmax(-1).type(torch.LongTensor).to(device)

        _, steps = pgd_(
            model,
            images,
            predicted,
            eps=epsilon,
            step=1 / 16,
            iters=25,
            targeted=False,
            device=device,
            clip_min=0,
            clip_max=1,
            random_step=random_step,
            report_steps=True,
            project=False,
        )

        steps = [step.reshape(step.shape[0], -1) for step in steps]
        similarities = torch.zeros(images.shape[0], len(steps) - 1).to(device)
        with torch.no_grad():
            for i in range(0, len(steps) - 1):
                if sequential:
                    similarities[:, i] = cos(steps[i], steps[i + 1])
                else:
                    similarities[:, i] = cos(steps[0], steps[i + 1])
        result.append(similarities)
    if return_dict:
        return {"PGD collinearity": torch.cat(result, dim=0).mean(dim=1).detach().cpu().numpy()}
    return torch.cat(result, dim=0)
