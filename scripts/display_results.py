import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

FILE_NAMES = ['_aggregated_metrics.csv', '_benchmarks.csv', '_fgsm_pgd_cos.csv', '_gradient_information.csv', '_gradient_norm.csv', '_linearization_error.csv', '_pgd_collinearity.csv']

def aggregated_metrics(dir, file_name, save_dir):
    aggregated_data = pd.read_csv(os.path.join(dir, file_name), index_col='Model').drop(columns=['Epsilons Range', 'FGSM Accuracy - Range', 'Random Accuracy - Range'], errors='ignore')
    # reorder columns so that 0.03 comes before 0.06
    columns = aggregated_data.columns.tolist()
    columns[1], columns[2], columns[3], columns[4], columns[6], columns[7] = columns[2], columns[1], columns[4], columns[3], columns[7], columns[6]
    aggregated_data = aggregated_data[columns]
    aggregated_data.index = [item.replace('0.06', '16/255').replace('0.03', '8/255') for item in aggregated_data.index.tolist()]
    # group metrics together
    aggregated_data.columns=pd.MultiIndex.from_arrays([['Clean Accuracy', 'FGSM Accuracy', 'FGSM Accuracy', 'PGD Accuracy', 'PGD Accuracy', 'PGD Accuracy', 'SPSA Accuracy', 'SPSA Accuracy', 'Gradient Norm', 'FGSM PGD Cosine Similarity', 'FGSM PGD Cosine Similarity', 'Linearization Error', 'Linearization Error', 'PGD Collinearity', 'Gradient Information'],
                                     ['-', '8/255', '16/255', '8/255', '16/255', 'Unbounded', '8/255', '16/255', '-', '8/255', '16/255', '8/255', '16/255', '-', '-']])
    aggregated_data = aggregated_data.round(decimals=4)

    # replace STEP with Lipschitz
    index = list(aggregated_data.index)
    index[index.index('STEP')] = 'Lipschitz'
    aggregated_data.index = index

    table = aggregated_data.reset_index().to_latex(index=False)
    with open(os.path.join(save_dir, file_name[:-4] + '_table.tex'), "w") as text_file:
        text_file.write(table)
    
    
    metrics_columns = ['Gradient Norm', 'FGSM PGD Cosine Similarity', 'Linearization Error', 'PGD Collinearity', 'Gradient Information']
    gradient_masking_data = aggregated_data.drop(level=0, columns=metrics_columns)
    table = gradient_masking_data.reset_index().to_latex(index=False)
    with open(os.path.join(save_dir, file_name[:-22] + 'gradient_masking_table.tex'), "w") as text_file:
        text_file.write(table)
    
    metrics_data = aggregated_data[metrics_columns]
    relative_metrics_data = ((metrics_data / metrics_data.loc['PGD eps: 16/255']) * 100).astype(int)
    table = metrics_data.reset_index().to_latex(index=False)
    with open(os.path.join(save_dir, file_name[:-22] + 'metrics_table.tex'), "w") as text_file:
        text_file.write(table)
    
    table = relative_metrics_data.reset_index().to_latex(index=False)
    with open(os.path.join(save_dir, file_name[:-22] + 'relative_metrics_table.tex'), "w") as text_file:
        text_file.write(table)


def benchmarks(dir, file_name, save_dir):
    benchmarks_data = pd.read_csv(os.path.join(dir, file_name)).drop(columns=['Unnamed: 0'])
    # replace STEP with Lipschitz
    benchmarks_data.loc[benchmarks_data['Model'] == 'STEP', 'Model'] = 'Lipschitz'

    g = sns.FacetGrid(benchmarks_data, col="Model", col_wrap=5)
    g.map(sns.lineplot,"Epsilons Range", "FGSM Accuracy - Range", label='FGSM')
    g.map(sns.lineplot,"Epsilons Range", "Random Accuracy - Range", color='red', label='Random')
    g.axes[0].legend()
    g.savefig(os.path.join(save_dir, file_name[:-4] + "_plot.png"))

def fgsm_pgd_cos(dir, file_name, save_dir):
    fgsm_pgd = pd.read_csv(os.path.join(dir, file_name)).drop(columns=['Unnamed: 0'])
    # replace STEP with Lipschitz
    fgsm_pgd.loc[fgsm_pgd['Model'] == 'STEP', 'Model'] = 'Lipschitz'
    fgsm_pgd.columns = ['0.03', '0.06', 'Model']
    long_form_cos_dif_data = pd.melt(fgsm_pgd, id_vars=['Model'], var_name='epsilon')
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.barplot(ax=ax, data=long_form_cos_dif_data, y='Model', x='value', hue='epsilon')
    fig.savefig(os.path.join(save_dir, file_name[:-4] + "_plot.png"), bbox_inches='tight')

def gradient_information(dir, file_name, save_dir):
    grad_info = pd.read_csv(os.path.join(dir, file_name)).drop(columns=['Unnamed: 0'])
    # replace STEP with Lipschitz
    grad_info.loc[grad_info['Model'] == 'STEP', 'Model'] = 'Lipschitz'
    g = sns.FacetGrid(grad_info, col="Model", col_wrap=5, sharex=False, xlim=(-1, 1), ylim = (0, 500))
    g.map(sns.histplot, "Gradient Information")
    means = grad_info.groupby('Model', sort=False).mean()['Gradient Information'].tolist()
    medians = grad_info.groupby('Model', sort=False).median()['Gradient Information'].tolist()
    for mean, median, ax in zip(means, medians, g.axes):
        ax.set(xlabel="Mean: {:.3f}, Median: {:.3f}".format(mean, median))
    g.fig.tight_layout()
    g.savefig(os.path.join(save_dir, file_name[:-4] + "_plot.png"))

def gradient_norm(dir, file_name, save_dir):
    grad_norm = pd.read_csv(os.path.join(dir, file_name)).drop(columns=['Unnamed: 0'])
    # replace STEP with Lipschitz
    grad_norm.loc[grad_norm['Model'] == 'STEP', 'Model'] = 'Lipschitz'
    g = sns.FacetGrid(grad_norm, col="Model", col_wrap=5, sharex=False, xlim=(10e-11, 10e0))
    g.map(sns.histplot, "Gradient Norm", log_scale=True, binwidth=1.5e-1)
    means = grad_norm.groupby('Model', sort=False).mean()['Gradient Norm'].tolist()
    medians = grad_norm.groupby('Model', sort=False).median()['Gradient Norm'].tolist()
    for mean, median, ax in zip(means, medians, g.axes):
        ax.set(xlabel="Mean: {:.3f}, Median: {:.3f}".format(mean, median))
    g.fig.tight_layout()
    g.savefig(os.path.join(save_dir, file_name[:-4] + "_plot.png"))

def linearization_error(dir, file_name, save_dir):
    lin_error = pd.read_csv(os.path.join(dir, file_name)).drop(columns=['Unnamed: 0'])
    # replace STEP with Lipschitz
    lin_error.loc[lin_error['Model'] == 'STEP', 'Model'] = 'Lipschitz'
    lin_error.columns = ['0.03', '0.06', 'Model']
    long_form_lin_error_data = pd.melt(lin_error, id_vars=['Model'], var_name='epsilon')
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.barplot(ax=ax, data=long_form_lin_error_data, y='Model', x='value', hue='epsilon').set(xscale='log')
    fig.savefig(os.path.join(save_dir, file_name[:-4] + "_plot.png"), bbox_inches='tight')

def pgd_collinearity(dir, file_name, save_dir):
    pgd_col = pd.read_csv(os.path.join(dir, file_name)).drop(columns=['Unnamed: 0'])
    # replace STEP with Lipschitz
    pgd_col.loc[pgd_col['Model'] == 'STEP', 'Model'] = 'Lipschitz'
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.barplot(ax=ax, data=pgd_col, y='Model', x='PGD collinearity')
    fig.savefig(os.path.join(save_dir, file_name[:-4] + "_plot.png"), bbox_inches='tight')

FUNCTIONS = [aggregated_metrics, benchmarks, fgsm_pgd_cos, gradient_information, gradient_norm, linearization_error, pgd_collinearity]


def display_results(dir, skip_benchmarks=False):
    if os.path.isdir(os.path.join(dir, 'results')):
        raise Exception ("Results already displayed. Remove directory 'results' and run this script again")
    os.mkdir(os.path.join(dir, "results"))
    for dataset in tqdm(['Train', 'Test']):
        for function, file_name in tqdm(zip(FUNCTIONS, FILE_NAMES)):
            file_name = dataset + file_name
            if file_name in os.listdir(dir):
                function(dir, file_name, os.path.join(dir, "results"))

if __name__ == '__main__':
    arguments = sys.argv
    if len(arguments) < 2 or not os.path.isdir(sys.argv[1]):
        raise Exception("Provide the directory where the results were generated")
    if len(arguments) > 2:
        display_results(sys.argv[1], skip_benchmarks=sys.argv[2].lower()=='true')
    else:
        display_results(sys.argv[1])