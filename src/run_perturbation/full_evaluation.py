import pandas as pd
import sys
import os
sys.path += ["code/python", "/mnt", ".."]

from MYRETRIEVE.code.evaluating.evaluate import compute_measure


from glob import glob
from tqdm import tqdm

from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def continuous_ndcg(run, qrel, k):

    def _dcg(vec):
        return np.sum(np.divide(np.array(vec), np.log2(np.arange(1, vec.size + 1)+1)))

    top_k_run = run.sort_values('score', ascending=False).groupby('query_id').head(k)
    assesed_run = top_k_run[["query_id", "doc_id"]].merge(qrel, how='left').fillna(0)

    qrel_sorted = qrel.sort_values('relevance', ascending=False).groupby('query_id').head(k)
    dcg_assessed = assesed_run.groupby('query_id')["relevance"].agg(_dcg).reset_index().rename(columns={'relevance': 'dcg_assessed'})
    dcg_qrels  = qrel_sorted.groupby("query_id")["relevance"].agg(_dcg).reset_index().rename(columns={'relevance': 'dcg_qrels'})
    dcg_assessed = dcg_assessed.merge(dcg_qrels)
    dcg_assessed["value"] = dcg_assessed["dcg_assessed"]/dcg_assessed["dcg_qrels"]
    dcg_assessed["measure"] = f"nDCG@{k}"
    return dcg_assessed[["query_id", "measure", "value"]]

def par_eval(parallel_args):
    run, qrels, qrel_type = parallel_args

    dset = run["dataset"].iloc[0]
    if qrel_type=="eRAG" and dset == "wow":
        perf = pd.concat([continuous_ndcg(run, qrels, 10), continuous_ndcg(run, qrels, 50)])
        perf[["dataset", "ir_model"]] = [run.iloc[0]["dataset"], run.iloc[0]["ir_model"]]
    else:
        perf = compute_measure(run, qrels, ["AP", "nDCG@10", "nDCG@50", "P@50", "R@50", "RR"])
        perf[["dataset", "ir_model"]] = [run.iloc[0]["dataset"], run.iloc[0]["ir_model"]]

    return perf


if __name__ == "__main__":

    #import runs
    runs = []
    for f in tqdm(glob("data/modified_runs/*")):
        ds, ir, rt = f.split("/")[-1].split(".")[0].split("_")

        runs.append(pd.read_csv(f, usecols=[0, 1, 2], dtype={"query_id": str, "doc_id": str}))

        runs[-1][["dataset", "ir_model"]] = [ds, f"{ir}_{rt}"]

    #import orig qrels
    orig_qrels = {}
    for f in glob("data/qrels/orig/*"):
        ds = f.split("/")[-1].split(".")[0]
        orig_qrels[ds] = pd.read_csv(f, sep=" ", dtype={"query_id": str, "doc_id": str}, usecols=[0, 2, 3], header=None, names=["query_id", "doc_id", "relevance"])

    #import eRAG qrels
    eRAG_qrels = {}
    for f in glob("data/qrels/eRAG/*"):
        ds = f.split("/")[-1].split(".")[0]
        eRAG_qrels[ds] = pd.read_csv(f, dtype={"query_id": str, "doc_id": str})

    #import downstream performance
    down_perfs = []
    for f in tqdm(glob("data/downstream/**/*.csv", recursive=True)):
        ds, ir, rt = f.split("/")[-1].split(".")[0].split("_")
        down_perfs.append(pd.read_csv(f, dtype={"query_id": str}))
        down_perfs[-1][["dataset", "ir_model"]] = [ds, f"{ir}_{rt}"]

    down_perfs = pd.concat(down_perfs).rename(columns={"score": "down"}).drop(columns=["k"])




    with Pool(processes=os.cpu_count()) as pool:
        eRAG_perf = pd.concat(pool.map(par_eval, [[r, eRAG_qrels[r.iloc[0]["dataset"]], "eRAG"] for r in runs if r.iloc[0]["dataset"] in eRAG_qrels])).rename(columns={"value": "eRAG"})

    with Pool(processes=os.cpu_count()) as pool:
        orig_perf = pd.concat(pool.map(par_eval, [[r, orig_qrels[r.iloc[0]["dataset"]], "orig"] for r in runs if r.iloc[0]["dataset"] in orig_qrels])).rename(columns={"value": "orig"})


    full_perfs = down_perfs.merge(orig_perf).merge(eRAG_perf, how="left").fillna(0)
    avg_perfs = full_perfs.groupby(["dataset", "ir_model", "measure"])[["down", "orig", "eRAG"]].mean().reset_index()




    '''
    print("computing the performance measures using original qrels", flush=True)
    orig_perf = runs.groupby(grouping_vars)[run_vars + ["dataset"]].apply(lambda x: evaluate_run(x[run_vars], orig_qrels[x.iloc[0]["dataset"]])).reset_index().drop(columns="level_3")

    print("computing the performance measures using eRAG qrels", flush=True)
    eRAG_perf = runs.groupby(grouping_vars)[run_vars + ["dataset"]].apply(lambda x: evaluate_run(x[run_vars], eRAG_qrels[x.iloc[0]["dataset"]])).reset_index().drop(columns="level_3")
    '''


    pse = None
    sre = None
    for corrm in ["pearson", "spearman", "kendall"]:
        paperstyle_eval = full_perfs.groupby(["dataset", "ir_model", "measure"])[["down", "orig", "eRAG"]].corr(method=corrm)\
                                    .reset_index().rename(columns={"level_3":"target"}).drop(columns=["orig", "eRAG"])\
                                    .query("target!='down'")\
                                    .pivot(index=["dataset", "ir_model", "measure"], columns=["target"], values=["down"])

        paperstyle_eval.columns = paperstyle_eval.columns.droplevel()
        paperstyle_eval.columns.name = ""
        paperstyle_eval = paperstyle_eval.reset_index()
        paperstyle_eval = paperstyle_eval.rename(columns={"orig": f"orig_{corrm}", "eRAG": f"eRAG_{corrm}"})
        if pse is None: pse = paperstyle_eval
        else: pse = pse.merge(paperstyle_eval)


        systemrank_eval = avg_perfs.groupby(["dataset", "measure"])[["down", "orig", "eRAG"]].corr(method=corrm)\
                                   .reset_index().rename(columns={"level_2":"target"}).drop(columns=["orig", "eRAG"])\
                                   .query("target!='down'") \
                                   .pivot(index=["dataset", "measure"], columns=["target"], values=["down"])

        systemrank_eval.columns = systemrank_eval.columns.droplevel()
        systemrank_eval.columns.name = ""
        systemrank_eval = systemrank_eval.reset_index()
        systemrank_eval = systemrank_eval.rename(columns={"orig": f"orig_{corrm}", "eRAG": f"eRAG_{corrm}"})

        if sre is None: sre = systemrank_eval
        else: sre = sre.merge(systemrank_eval)


    # df columns: dataset, ir_model, measure, down, orig, eRAG

    datasets = list(avg_perfs["dataset"].unique())
    measures = list(avg_perfs["measure"].unique())


    #plot the performance
    tmp_avg_perfs = avg_perfs.copy()
    tmp_avg_perfs = tmp_avg_perfs.query("dataset=='nq'")
    tmp_avg_perfs[["ir_model", "run_type"]] = tmp_avg_perfs.ir_model.str.split("_", expand=True)
    tmp_avg_perfs = tmp_avg_perfs.query("measure=='nDCG@50'")[["ir_model", "run_type", "orig", "down"]].sort_values("down", ascending=True)
    tmp_avg_perfs["run_number"] = np.arange(len(tmp_avg_perfs))

    # Increase global font sizes
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    fig = plt.figure(figsize=(12, 4))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    sns.scatterplot(
        data=tmp_avg_perfs,
        y="orig",
        x="run_number",
        hue="ir_model",
        style="run_type",
        s=120,  # marker size (points^2)
    )

    plt.savefig("data/figures/performance_nq.pdf", bbox_inches="tight")

    tmp_avg_perfs = avg_perfs.copy()
    tmp_avg_perfs = tmp_avg_perfs.query("dataset=='nq'")
    tmp_avg_perfs[["ir_model", "run_type"]] = tmp_avg_perfs.ir_model.str.split("_", expand=True)

    tmp_avg_perfs = tmp_avg_perfs.query("measure=='nDCG@50'")[["ir_model", "run_type", "down", "orig"]].sort_values("down", ascending=True)
    tmp_avg_perfs["run_number"] = np.arange(len(tmp_avg_perfs))

    # Increase global font sizes
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    fig = plt.figure(figsize=(12, 4))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    sns.scatterplot(
        data=tmp_avg_perfs,
        y="down",
        x="run_number",
        hue="ir_model",
        style="run_type",
        s=120,  # marker size (points^2)
    )
    plt.savefig("data/figures/down_performance_nq.pdf", bbox_inches="tight")

    tmp_avg_perfs = avg_perfs.copy()
    tmp_avg_perfs = tmp_avg_perfs.query("dataset=='nq'")
    tmp_avg_perfs[["ir_model", "run_type"]] = tmp_avg_perfs.ir_model.str.split("_", expand=True)

    tmp_avg_perfs = tmp_avg_perfs.query("measure=='nDCG@50'")[["ir_model", "run_type", "eRAG", "orig", "down"]].sort_values("down", ascending=True)
    tmp_avg_perfs["run_number"] = np.arange(len(tmp_avg_perfs))

    # Increase global font sizes
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    fig = plt.figure(figsize=(12, 4))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    sns.scatterplot(
        data=tmp_avg_perfs,
        y="eRAG",
        x="run_number",
        hue="ir_model",
        style="run_type",
        s=120,  # marker size (points^2)
    )
    plt.savefig("data/figures/eRAG_performance_nq.pdf", bbox_inches="tight")

    nrows, ncols = len(datasets), len(measures)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for i, ds in enumerate(datasets):
        for j, ms in enumerate(measures):
            ax = axes[i, j]
            sub = avg_perfs[(avg_perfs["dataset"] == ds) & (avg_perfs["measure"] == ms)]

            if sub.empty:
                ax.set_axis_off()
                continue

            # red crosses: down vs orig
            ax.scatter(sub["orig"], sub["down"], marker="x", label="orig", alpha=0.9)

            # blue bullets: down vs eRAG
            ax.scatter(sub["eRAG"], sub["down"], marker="o", label="eRAG", alpha=0.7)

            # Titles/labels laid out like a mosaic: row=dataset, col=measure
            if i == 0:
                ax.set_title(ms)
            if j == 0:
                ax.set_ylabel(f"{ds}\n\ndownstream performance")
            if i == nrows - 1:
                ax.set_xlabel("score")

            ax.grid(True, alpha=0.3)

    # one legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="x", color="red", linestyle="None", label="orig"),
        Line2D([0], [0], marker="o", color="blue", linestyle="None", label="eRAG"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
