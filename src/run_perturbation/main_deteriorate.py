import os

import pytrec_eval

os.environ['IR_DATASETS_HOME'] = '/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/ir_datasets'

import ir_datasets
import pandas as pd
from tqdm import tqdm
import sys
sys.path += ["code", "code/python"]

from deteriorate_run import deteriorate_run

from ir_measures import nDCG
import ir_measures



pairs = [(ds, ir) for ds in ["nq", "fever", "wow"] for ir in ["bge", "bm25", "contriever", "dpr", "tct"]]

for ds, ir in tqdm(pairs):
    run = pd.read_csv(f"data/runs/{ds}/{ir}.txt", header=None,
                      usecols=[0, 2, 4], names=["query_id", "doc_id", "score"], sep=" ", dtype={"query_id": str, "doc_id": str, "score": float})
    qrels = pd.read_csv(f"data/qrels/{ds}.txt", header=None, usecols=[0,2, 3], names=["query_id", "doc_id", "relevance"], sep=" ",
                        dtype={"query_id": str, "doc_id": str, "relevance": int})

    params = [('verymuchworse', 'worse', 25, 25), ('muchworse', 'worse', 15, 15), ('worse', 'worse', 3, 3), ('better', 'better', 3, 3), ('muchbetter', 'better', 10, 10), ('verymuchbetter', 'better', 20, 20)]
    for p in params:
        name, mode, number_swaps, number_replacements = p
        out = deteriorate_run(f"data/runs/{ds}/{ir}.txt",
                            f"data/qrels/{ds}.txt",
                            1,
                             (15, 25), (26, 51),
                            mode, number_swaps, number_replacements,
                            False)

        new_run =  pd.DataFrame(
            [(k, k2, v) for k, inner in out.items() for k2, v in inner.items()],
            columns=["query_id", "doc_id", "score"]
        )
        print("\n\n")
        print(f"{ds} {ir}: {ir_measures.calc_aggregate([nDCG@50], qrels, run)[nDCG@50]}")
        print(f"{ds} {ir}: {ir_measures.calc_aggregate([nDCG@50], qrels, new_run)[nDCG@50]}")

        new_run["run_id"] = f"{ds}_{ir}_better"
        new_run.to_csv(f"data/modified_runs/{ds}_{ir}_{name}.csv", index=False)