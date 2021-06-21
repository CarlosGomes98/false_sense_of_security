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
from robustbench.utils import load_model
import torchvision.models as torchvision_models
from src.trainers import Trainer, FGSMTrainer
from src.utils import adversarial_accuracy, fgsm_, pgd_, plot_along_grad_n
from src.load_architecture import CIFAR_Wide_Res_Net, CIFAR_Res_Net, CIFAR_Net, CUREResNet18, StepResNet18
from src.gradient_masking_tests import (
    run_masking_benchmarks,
    get_accuracy,
    pgd_collinearity,
    fgsm_pgd_cos_dif,
    linearization_error,
    gradient_norm,
    gradient_information,
)


def generate_results(models, metrics, dir, device="cpu", save_raw_data=True):
    # setup
    device = torch.device(device)
    print("Running on {}".format(device))
    batch_size = 128
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    for dataset_name, dataset in tqdm(zip(["Train", "Test"], [train_dataset, test_dataset])):

        results = []
        for model_name, model in tqdm(models.items()):
            print("\nRunning model {}:".format(model_name))
            results_dict = {'Model': model_name}
            for metric_name, metric in tqdm(metrics.items()):
                print("\n\tRunning metric {}".format(metric_name))
                res = metric(
                    model, dataset, return_dict=True, batch_size=batch_size, device=device, subset_size=5000)
                results_dict[metric_name] = res
            results.append(results_dict)
        save_data_and_overview(results, dir, dataset_name, save_raw_data, list(metrics.keys()))

def save_data_and_overview(results, dir, dataset_name, save_raw_data, metrics):
    # go through the metrics and save the raw results to a different file, if save_raw_data is True
    # benchmarks and model name ignored here
    # also flatten out results

    metrics_dataframes = {metric_name: None for metric_name in metrics if metric_name != 'Model'}
    aggregated_table = []
    for result in results:
        aggregated = {'Model': result['Model']}
        for metric_group, metric in result.items():
            if metric_group == 'Model':
                continue
            # dont save numpy arrays to pd dataframe. put in dif file.
            long_form_metric_group = {}
            for name, res in metric.items():
                if isinstance(res, np.ndarray):
                    aggregated[name] = res.mean()
                    long_form_metric_group[name] = res
                else:
                    aggregated[name] = res

            if len(long_form_metric_group) != 0 and save_raw_data:
                metric_df = pd.DataFrame(data=long_form_metric_group)
                metric_df['Model'] = aggregated['Model']
                if metrics_dataframes[metric_group] is None:
                    metrics_dataframes[metric_group] = metric_df
                else:
                    metrics_dataframes[metric_group] = pd.concat([metrics_dataframes[metric_group], metric_df])
        
        aggregated_table.append(aggregated)
    
    aggregated_table = pd.DataFrame(data=aggregated_table)
    aggregated_table.set_index('Model')
    aggregated_table.to_csv(os.path.join(dir, dataset_name + '_aggregated_metrics.csv'), index=False)

    if save_raw_data:
        for metric_name, df in metrics_dataframes.items():
            if df is None: 
                continue
            df.to_csv(os.path.join(dir, dataset_name + '_' + metric_name + '.csv'))

if __name__ == "__main__":
    model_names = [
        "Normal",
        "Step-ll eps: 8/255",
        "Step-ll eps: 16/255",
        "FGSM eps: 8/255",
        "FGSM eps: 8/255 (catastrophic overfitting)",
        "FGSM eps: 16/255",
        "PGD eps: 8/255",
        "PGD eps: 16/255",
        "Jacobian Regularization 0.1",
        "Jacobian Regularization 0.5",
        "Jacobian Regularization 1",
        "Lipschitz",
        "CURE",
        "Adversarial_Interpolation",
        "Feature_Scatter",
        "Pre_Training"
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
        "--models",
        default=None,
        nargs='*',
        help="Models to run metrics on. If flag not used will run on all models.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        choices=list(all_metrics.keys()),
        help="Metric to execute. If flag not used will execute all metrics",
    )
    
    parser.add_argument(
        "--skip_benchmarks", action="store_true", help="Skip the benchmarks, which take a while"
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
        metrics = dict(all_metrics)
        if args.skip_benchmarks:
            del metrics["benchmarks"]
    else:
        metrics = {args.metric: all_metrics[args.metric]}

    device = torch.device(args.device)
    # # Regular CIFAR-10  ResNet Model
    model = CIFAR_Res_Net().eval().to(device)
    # model.load_state_dict(torch.load("models/normal_20e.model", map_location=device))
    # CIFAR-10  ResNet Model trained with pgd 03
    pgd_model = CIFAR_Res_Net().to(device).eval()
    # pgd_model.load_state_dict(
        # torch.load("models/pgd_e8_20e.model", map_location=device)
    # )
    # CIFAR-10  ResNet Model trained with pgd 06
    pgd_model_6 = CIFAR_Res_Net().to(device).eval()
    # pgd_model_6.load_state_dict(
        # torch.load("models/pgd_e16_20e.model", map_location=device)
    # )
    # CIFAR-10  ResNet Model trained with large FGSM steps
    fgsm_model = CIFAR_Res_Net().to(device).eval()
    # fgsm_model.load_state_dict(
    #     torch.load("models/fgsm_e16_20e.model", map_location=device)
    # )
    # # # # CIFAR-10  ResNet Model trained with small FGSM steps (grad mask)
    fgsm_model_small = CIFAR_Res_Net().to(device).eval()
    # fgsm_model_small.load_state_dict(
    #     torch.load("models/fgsm_e8_20e.model", map_location=device)
    # )
    # # # # CIFAR-10  ResNet Model trained with small FGSM steps (no grad mask)
    fgsm_model_small_2 = CIFAR_Res_Net().to(device).eval()
    # fgsm_model_small_2.load_state_dict(
    #     torch.load("models/fgsm_e8_20e_2.model", map_location=device)
    # )
    # # # CIFAR-10  ResNet Model trained with large Step-ll steps
    step_ll_model = CIFAR_Res_Net().to(device).eval()
    # step_ll_model.load_state_dict(
    #     torch.load("models/step_ll_e16_20e.model", map_location=device)
    # )
    # # # CIFAR-10  ResNet Model trained with small Step-ll steps
    step_ll_model_small = CIFAR_Res_Net().to(device).eval()
    # step_ll_model_small.load_state_dict(
    #     torch.load("models/step_ll_e8_20e.model", map_location=device)
    # )

    # # CIFAR-10  ResNet Model trained through Jacobian regularization ld0.5
    jac_regularization_model_005 = CIFAR_Res_Net().to(device).eval()
    # jac_regularization_model_005.load_state_dict(
    #     torch.load("models/jac_regularization_ld005_20.model", map_location=device)
    # )

    jac_regularization_model_01 = CIFAR_Res_Net().to(device).eval()
    # jac_regularization_model_01.load_state_dict(
    #     torch.load("models/jac_regularization_ld01_20.model", map_location=device)
    # )

    jac_regularization_model_05 = CIFAR_Res_Net().to(device).eval()
    # jac_regularization_model_05.load_state_dict(
    #     torch.load("models/jac_regularization_ld05_20.model", map_location=device)
    # )
    
    jac_regularization_model_1 = CIFAR_Res_Net().to(device).eval()
    # jac_regularization_model_1.load_state_dict(
    #     torch.load("models/jac_regularization_ld1_20.model", map_location=device)
    # )
    # Step
    step = StepResNet18().to(device).eval()
    # step.load_state_dict(
    #     torch.load("models/rn18_std_step_convergence1.pt", map_location=device)['model_state_dict'])
    
    ## Pretrained CIFAR-10 RESNET trained using CURE
    cure = CUREResNet18().to(device).eval()
    # cure[1].load_state_dict(
    #     torch.load("models/RN18_CURE.pth", map_location=device)["net"]
    # )
    
    #advInterp
    adv_interp_weights = torch.load("models/advInterp", map_location=device)['net']
    adv_interp_weights_fixed = {key[7:]: val for (key, val) in adv_interp_weights.items()}
    adv_interp = CIFAR_Wide_Res_Net().eval().to(device)
    adv_interp.load_state_dict(adv_interp_weights_fixed)

    #featureScatter
    feature_scatter_weights = torch.load("models/featureScatter", map_location=device)['net']
    feature_scatter_weights_fixed = {key[17:]: val for (key, val) in feature_scatter_weights.items()}
    feature_scatter = CIFAR_Wide_Res_Net().eval().to(device)
    feature_scatter.load_state_dict(feature_scatter_weights_fixed)

    # #hendrycks
    # hendrycks_weights = torch.load("D:\Libraries\Downloads\cifar10wrn_baseline_epoch_4.pt", map_location=device)
    # hendrycks_weights_fixed = {key.replace("module.", ""): val for (key, val) in hendrycks_weights.items()}
    # hendrycks_model = CIFAR_Wide_Res_Net().eval().to(device)
    # hendrycks_model.load_state_dict(hendrycks_weights_fixed)

    full_resnet_weights = torch.load("../models/resnet_full_train.model", map_location=device)['net']
    full_resnet_weights_fixed = {key.replace("module.", ""): val for (key, val) in full_resnet_weights.items()}
    full_resnet = CIFAR_Res_Net(normalization_mean=[0.4914, 0.4822, 0.4465], normalization_std=[0.2023, 0.1994, 0.2010]).to(device).eval()
    full_resnet.load_state_dict(
        full_resnet_weights_fixed
    )


    models = [
        model,
        step_ll_model_small,
        step_ll_model,
        fgsm_model_small_2,
        fgsm_model_small,
        fgsm_model,
        pgd_model,
        pgd_model_6,
        jac_regularization_model_01,
        jac_regularization_model_05,
        jac_regularization_model_1,
        step,
        cure,
        adv_interp,
        feature_scatter
    ]

    all_models = dict(zip(model_names, models))
    if args.models is None:
        models = all_models
    else:
        models = {}
        for model_name in args.models:
            if model_name in model_names:
                models[model_name] = all_models[model_name]
            else:
                try:
                    models[model_name] = load_model(model_name=model_name, dataset='cifar10', model_dir='models', threat_model='Linf').to(device)
                except:
                    raise Exception("Model {} cannot be loaded".format(model_name))
    generate_results(models, metrics, args.dir, device=device)
