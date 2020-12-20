import tqdm
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
from src.gradient_masking_tests import run_masking_benchmarks, get_accuracy, pgd_colinearity

# setup
device = torch.device("cpu")
batch_size = 32
# remove the normalize
transform = transform = transforms.Compose(
            [transforms.ToTensor()]
)
        
normalized_min = (0 - 0.5) / 0.5
normalized_max = (1 - 0.5) / 0.5
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=2)
classes = classes = ('plane', 'car', 'bird', 'cat',
   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = CIFAR_Res_Net().eval().to(device)
model.load_state_dict(torch.load("./models/normal_20e.model", map_location=device))
fgsm_model = CIFAR_Res_Net().eval().to(device)
fgsm_model.load_state_dict(torch.load("./models/fgsm_e16_20e.model", map_location=device))
fgsm_model_small = CIFAR_Res_Net().eval().to(device)
fgsm_model_small.load_state_dict(torch.load("./models/fgsm_e8_20e.model", map_location=device))
step_ll_model = CIFAR_Res_Net().eval().to(device)
step_ll_model.load_state_dict(torch.load("./models/step_ll_e16_20e.model", map_location=device))
step_ll_model_small = CIFAR_Res_Net().eval().to(device)
step_ll_model_small.load_state_dict(torch.load("./models/step_ll_e8_20e.model", map_location=device))
pgd_model = CIFAR_Res_Net().eval().to(device)
pgd_model.load_state_dict(torch.load("./models/pgd_e16_20e.model", map_location=device))
pgd_model_small = CIFAR_Res_Net().eval().to(device)
pgd_model_small.load_state_dict(torch.load("./models/pgd_e8_20e.model", map_location=device))
grad_reg_model = CIFAR_Res_Net().eval().to(device)
grad_reg_model.load_state_dict(torch.load("./models/grad_reg_ld01_20e.model", map_location=device))
cure = CUREResNet18().to(device).eval()
cure[1].load_state_dict(torch.load("./models/RN18_CURE.pth", map_location=device)['net'])

models = [model, fgsm_model, fgsm_model_small, pgd_model, pgd_model_small, step_ll_model, step_ll_model_small, grad_reg_model, cure]
model_names = ['normal', 'fgsm', 'fgsm small', 'pgd', 'pgd small', 'step_ll', 'step_ll small', 'grad reg', 'cure']
data = [pd.DataFrame(pgd_colinearity(m, train_dataset, 0.03, device=device, random_step=True, subset_size=5000, sequential=True).detach().cpu().numpy()) for m in models]
for data_item, model_name in zip(data, model_names):
   data_item['sum'] = data_item.sum(axis=1)
   data_item['mean'] = data_item['sum'] / (data_item.shape[1] - 1)
   data_item['model'] = model_name

data = pd.concat(data)
data.to_csv("pgd_colinearity.csv")

data = [pd.DataFrame(pgd_colinearity(m, train_dataset, 0.03, device=device, random_step=True, subset_size=5000, sequential=False).detach().cpu().numpy()) for m in models]
for data_item, model_name in zip(data, model_names):
   data_item['sum'] = data_item.sum(axis=1)
   data_item['mean'] = data_item['sum'] / (data_item.shape[1] - 1)
   data_item['model'] = model_name

data = pd.concat(data)
data.to_csv("pgd_colinearity_non_sequential.csv")
# data = pd.DataFrame(data=torch.stack([pgd_colinearity(m, train_dataset, 0.03, device=device, random_step=True).detach().cpu() for m in models], dim=1).numpy(), columns=model_names)
# fig, ax = plt.subplots(figsize=(15, 12))
# sns.barplot(ax=ax, data=data)
# ax.set(ylabel='PGD Colinearity score', title='PGD Colinearity epsilon 0.03(8/256) test set')
# fig.savefig("pgd_colinearity_plot_03_random_step_train.png")

