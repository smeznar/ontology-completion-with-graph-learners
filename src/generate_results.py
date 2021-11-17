import pandas as pd
import glob
import numpy as np


if __name__ == '__main__':
    order = ["marine", "anatomy", "scto", "emotions", "ehdaa", "foodon", "LKN", "go"]
    baselines = []
    datasets = []
    rocs = []
    aps = []
    for fn in glob.glob("../results/*"):
        with open(fn, "r") as file:
            a = file.readline().split("\t")
            baselines.append(a[0])
            datasets.append(a[1].strip())
            a = [float(i)*100 for i in file.readline().split("\t")]
            rocs.append("{} ($\pm$ {})".format(np.round(np.mean(a), 2), np.round(np.std(a), 2)))
            a = [float(i)*100 for i in file.readline().split("\t")]
            aps.append("{} ($\pm$ {})".format(np.round(np.mean(a), 2), np.round(np.std(a), 2)))
    df = pd.DataFrame(data={"ROC": rocs, "Baseline": baselines, "Dataset": datasets, "AP": aps})
    df["Dataset"] = pd.Categorical(df["Dataset"], order)
    df.sort_values(by='Dataset')
    df_ap = df.pivot_table(values="AP", columns="Dataset", index="Baseline", aggfunc=lambda x: ' '.join(x))
    df_ap.to_latex("ap.tex", escape=False)
    df_roc = df.pivot_table(values="ROC", columns="Dataset", index="Baseline", aggfunc=lambda x: ' '.join(x))
    df_roc.to_latex("roc.tex", escape=False)
