import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FILE_LOCATION = "metrics_less_samples_more_steps_same_sample"
for dataset in ["Train_", "Test_"]:
    path = os.path.join(FILE_LOCATION, dataset + "aggregated_metrics.csv")
    data = pd.read_csv(path, index_col="Model")

    robustbench_names = [
        "Standard",
        "Gowal2020Uncovering_28_10_extra",
        "Wu2020Adversarial_extra",
        "Carmon2019Unlabeled",
        "Wang2020Improving",
        "Zhang2020Geometry",
        "Wong2020Fast",
        "Hendrycks2019Using",
        "Sehwag2020Hydra",
        "Adversarial Interpolation",
        "Feature Scatter",
        "Hendrycks2019Using",
        "Sehwag2020Hydra"
    ]

    data = data.loc[robustbench_names]
    data["FGSM-SPSA eps: 8/255"] = (
        data["FGSM Accuracy eps: 0.03"] - data["SPSA Accuracy eps: 0.03"]
    )
    data["FGSM-SPSA eps: 16/255"] = (
        data["FGSM Accuracy eps: 0.06"] - data["SPSA Accuracy eps: 0.06"]
    )
    data["FGSM-PGD eps: 8/255"] = (
        data["FGSM Accuracy eps: 0.03"] - data["PGD Accuracy eps: 0.03"]
    )
    data["FGSM-PGD eps: 16/255"] = (
        data["FGSM Accuracy eps: 0.06"] - data["PGD Accuracy eps: 0.06"]
    )

    print(data.columns)
    data = data[
        [
            "Gradient Norm",
            "FGSM PGD Cosine Similarity eps: 0.03",
            "FGSM PGD Cosine Similarity eps: 0.06",
            "Linearization Error eps: 0.03",
            "Linearization Error eps: 0.06",
            "PGD collinearity",
            "Gradient Information",
            "FGSM-SPSA eps: 8/255",
            "FGSM-SPSA eps: 16/255",
            "FGSM-PGD eps: 8/255",
            "FGSM-PGD eps: 16/255",
        ]
    ]

    data.index = [
        item.replace("0.06", "16/255").replace("0.03", "8/255")
        for item in data.index.tolist()
    ]
    data.columns = [
        item.replace("0.06", "16/255")
        .replace("0.03", "8/255")
        .replace("Gradient Information", "Robustness Information")
        for item in data.columns.tolist()
    ]

    metrics_corr = data[
        [
            "Gradient Norm",
            "FGSM PGD Cosine Similarity eps: 8/255",
            "FGSM PGD Cosine Similarity eps: 16/255",
            "Linearization Error eps: 8/255",
            "Linearization Error eps: 16/255",
            "PGD collinearity",
            "Robustness Information",
        ]
    ].corr()

    mask = np.triu(np.ones_like(metrics_corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(metrics_corr, cmap="vlag", annot=True, mask=mask)
    fig.savefig(
        os.path.join(FILE_LOCATION, dataset + "metrics_corr_papers.png"), bbox_inches="tight"
    )

    corr = data.corr()[
        [
            "FGSM-SPSA eps: 8/255",
            "FGSM-SPSA eps: 16/255",
            "FGSM-PGD eps: 8/255",
            "FGSM-PGD eps: 16/255",
        ]
    ].sort_values(by="FGSM-SPSA eps: 8/255", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="vlag", annot=True)
    fig.savefig(
        os.path.join(FILE_LOCATION, dataset + "results_corr_papers.png"), bbox_inches="tight"
    )
