import os
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, FGSM
from advertorch.attacks import LinfSPSAAttack
from src.trainers import Trainer, FGSMTrainer
from src.utils import adversarial_accuracy, fgsm_, pgd_, plot_along_grad_n
from src.Nets import CIFAR_Wide_Res_Net, CIFAR_Res_Net, CIFAR_Net, CUREResNet18
from src.gradient_masking_tests import (
    run_masking_benchmarks,
    get_accuracy,
    pgd_collinearity,
    fgsm_pgd_cos_dif,
    linearization_error,
    gradient_norm,
    gradient_information,
)


def generate_results(models, metrics, dir, device="cpu"):
    # setup
    device = torch.device(device)
    batch_size = 128
    # remove the normalize
    transform = transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    for dataset_name, dataset in tqdm(zip(["Train", "Test"], [train_dataset, test_dataset])):

        results = []
        for model_name in tqdm(models):
            model = models[model_name]

            results_dict = {"Model": model_name}
            for metric_name in tqdm(metrics):
                res = metrics[metric_name](
                    model, dataset, return_dict=True, subset_size=2, batch_size=batch_size, device=device
                )
                for r in res:
                    results_dict[r] = res[r]
            results.append(results_dict)
        results_df = pd.DataFrame(data=results)
        results_df.set_index("Model")
        results_df.to_csv(os.path.join(dir, dataset_name + "_metrics.csv"))


if __name__ == "__main__":
    model_names = [
        "Normal",
        "Step ll eps: 0.06",
        "Step ll eps: 0.03",
        "FGSM eps: 0.06",
        "Fgsm eps: 0.03 (catastrophic overfitting)",
        "FGSM eps: 0.03",
        "PGD eps: 0.06",
        "PGD eps: 0.03",
        "Jacobian Regularization 0.1",
        "Jacobian Regularization 0.2",
        "Jacobian Regularization 0.5",
        "CURE",
    ]
    all_metrics = {
        "benchmarks": run_masking_benchmarks,
        "gradient_norm": gradient_norm,
        "fgsm_pgd_cos": fgsm_pgd_cos_dif,
        "linearization_error": linearization_error,
        "pgd_collinearity": pgd_collinearity,
        "gradient_information": gradient_information,
    }

    parser = argparse.ArgumentParser(
        description="Run a set of metrics on a set of models"
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=model_names,
        help="Model to run metrics on. If flag not used will run on all models.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        choices=list(all_metrics.keys()),
        help="Metric to execute. If flag not used will execute all metrics",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Also save results in plots and tables"
    )
    parser.add_argument("--dir", type=str, help="Directory where results will be stored", required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Execute on cpu or gpu",
    )

    args = parser.parse_args()

    if os.path.isdir(args.dir):
        raise Exception("Directory already exists")

    os.mkdir(args.dir)
    if args.metric is None:
        metrics = all_metrics
    else:
        metrics = {args.metric: all_metrics[args.metric]}

    device = torch.device(args.device)
    # # Regular CIFAR-10  ResNet Model
    model = CIFAR_Res_Net().eval().to(device)
    model.load_state_dict(torch.load("models/normal_20e.model", map_location=device))
    # CIFAR-10  ResNet Model trained with pgd 03
    pgd_model = CIFAR_Res_Net().to(device).eval()
    pgd_model.load_state_dict(
        torch.load("models/pgd_e8_20e.model", map_location=device)
    )
    # CIFAR-10  ResNet Model trained with pgd 06
    pgd_model_6 = CIFAR_Res_Net().to(device).eval()
    pgd_model_6.load_state_dict(
        torch.load("models/pgd_e16_20e.model", map_location=device)
    )
    # CIFAR-10  ResNet Model trained with large FGSM steps
    fgsm_model = CIFAR_Res_Net().to(device).eval()
    fgsm_model.load_state_dict(
        torch.load("models/fgsm_e16_20e.model", map_location=device)
    )
    # # # # CIFAR-10  ResNet Model trained with small FGSM steps (grad mask)
    fgsm_model_small = CIFAR_Res_Net().to(device).eval()
    fgsm_model_small.load_state_dict(
        torch.load("models/fgsm_e8_20e.model", map_location=device)
    )
    # # # # CIFAR-10  ResNet Model trained with small FGSM steps (no grad mask)
    fgsm_model_small_2 = CIFAR_Res_Net().to(device).eval()
    fgsm_model_small_2.load_state_dict(
        torch.load("models/fgsm_e8_20e_2.model", map_location=device)
    )
    # # # CIFAR-10  ResNet Model trained with large Step-ll steps
    step_ll_model = CIFAR_Res_Net().to(device).eval()
    step_ll_model.load_state_dict(
        torch.load("models/step_ll_e16_20e.model", map_location=device)
    )
    # # # CIFAR-10  ResNet Model trained with small Step-ll steps
    step_ll_model_small = CIFAR_Res_Net().to(device).eval()
    step_ll_model_small.load_state_dict(
        torch.load("models/step_ll_e8_20e.model", map_location=device)
    )
    # # CIFAR-10  ResNet Model trained through Jacobian regularization ld0.1
    jac_norm_model = CIFAR_Res_Net().to(device).eval()
    jac_norm_model.load_state_dict(
        torch.load("models/grad_reg_ld01_20e.model", map_location=device)
    )
    # # CIFAR-10  ResNet Model trained through Jacobian regularization ld0.2
    jac_norm_model_2 = CIFAR_Res_Net().to(device).eval()
    jac_norm_model_2.load_state_dict(
        torch.load("models/grad_reg_ld02_20e.model", map_location=device)
    )
    ## Pretrained CIFAR-10 RESNET trained using CURE
    cure = CUREResNet18().to(device).eval()
    cure[1].load_state_dict(
        torch.load("models/RN18_CURE.pth", map_location=device)["net"]
    )

    models = [
        model,
        step_ll_model,
        step_ll_model_small,
        fgsm_model,
        fgsm_model_small,
        fgsm_model_small_2,
        pgd_model_6,
        pgd_model,
        jac_norm_model,
        jac_norm_model_2,
        cure,
    ]

    all_models = dict(zip(model_names, models))
    if args.model is None:
        models = all_models
    else:
        models = {args.model: all_models[args.model]}
    generate_results(models, metrics, args.dir, device=device)
