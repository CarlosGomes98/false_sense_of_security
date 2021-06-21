import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# maybe arch differences dont matter so much. add the other models too. wtv.
AA_dif = {
    'Gowal2020Uncovering_28_10_extra': {
        'pgd': 67.9,
        'aa': 62.8
    },
    'Wu2020Adversarial_extra': {
        'pgd': 67,
        'aa': 60
    },
    'Carmon2019Unlabeled': {
        'pgd': 66.3,
        'aa': 59.5
    },
    'Wang2020Improving': {
        'pgd': 67.1,
        'aa': 56
    },
    # CURE has a dif architecture (resnet18)
    'CURE': {
        'pgd': 42.4,
        'aa': 38.5
    },
    # zhang the reported is for a different architecture
    'Zhang2020Geometry': {
        'pgd': 59.6,
        'aa': 55.6
    },
    # this is a resnet18 as well
    'Wong2020Fast': {
        'pgd': 50.6,
        'aa': 43.2
    },
    'Feature_Scatter': {
        'pgd': 75.3,
        'aa': 36.6
    },
    'Adversarial_Interpolation': {
        'pgd': 76.6,
        'aa': 36.4
    },
    'Hendrycks2019Using': {
        'pgd': 61.2,
        'aa': 54.9
    },
    'Sehwag2020Hydra': {
        'pgd': 64.4,
        'aa': 57.1
    }
}

metrics = ['Gradient Norm',
       'FGSM PGD Cosine Similarity eps: 0.03',
       'FGSM PGD Cosine Similarity eps: 0.06', 'Linearization Error eps: 0.03',
       'Linearization Error eps: 0.06', 'PGD collinearity',
       'Gradient Information']

dir = 'full_results_really_now_more'
for dataset in ['Train', 'Test']:
    table = pd.read_csv('{}/{}_aggregated_metrics.csv'.format(dir, dataset), index_col='Model')
    standard_scores = table[metrics].loc['Standard']
    table = table[metrics].loc[list(AA_dif.keys())]
    AA_dif_metric = {key: (value['pgd'] - value['aa'])/ value['aa'] for key, value in AA_dif.items()}
    table['AA_dif'] = pd.Series(AA_dif_metric)
    for metric in metrics:
        metric_table = table[[metric, 'AA_dif']]
        # print(metric_table)
        plt.figure(figsize=(10,6))
        plot = sns.scatterplot(data=metric_table, x=metric, y='AA_dif', hue='Model')
        # uncomment to draw line for standard model. Messes up the scale
        # plt.axvline(standard_scores[metric], table[['AA_dif']].min(), table[['AA_dif']].max())
        plt.title("PGD - Auto Attack vs " + metric)
        plt.tight_layout()
        print('{}/results/aa_normalized_'.format(dir) + metric.replace(':', '').replace('.', '') + '_AA_' + dataset + '.png')
        plt.savefig('{}/results/aa_normalized_'.format(dir) + metric.replace(':', '').replace('.', '') + '_AA_' + dataset + '.png')
        plt.clf()