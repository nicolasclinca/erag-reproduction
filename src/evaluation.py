"""
evaluation.py
Valutazione pipeline RAG (retrieval + generazione T5-FiD):
- Metriche eRAG
- Punteggi end-to-end
- Correlazioni Spearman/Kendall tra metriche eRAG e performance end-to-end
Salva log JSON in ../logs

Uso CLI:
  python evaluation.py --model_dir ../models/fid_t5 \
                       --k_values 10 30 50 \
                       --method BM25 \
                       --test_dataset_path ../data/nq-dev-kilt.jsonl \
                       --metric em
"""

import json
import re
import torch
import erag_mod
import os
import scipy.stats as stats
from functools import partial
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_loader import retrieval_results, load_expected_outputs
import argparse
from fid_t5 import t5_fid_generator
from contriever_retriever import DenseRetriever
from metrics import exact_match_metric, f1_metric


METRICS = {
    "em": exact_match_metric,
    "f1": f1_metric,
    "accuracy": exact_match_metric # In our case exact match and accuracy coincide
}

def save_json_log(data, file_path, description=""):
    """
    Salva un dizionario in JSON e stampa un messaggio di conferma.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"{description} saved in {file_path}.")

def _limit_docs_per_query(retrieval_results_dict, k):
    """
    Ritorna una copia di retrieval_results_dict dove per ogni query i documenti
    sono limitati ai primi k.
    """
    return {q: docs[:k] for q, docs in retrieval_results_dict.items()}


def evaluation_erag(
    expected_outputs,
    retrieval_results_dict,
    t5_generator_for_eval,
    downstream_metric_func,
    retrieval_metrics,
    method,
    log_dir="../logs",
):

    print(f"\nEvaluating eRAG scores...")

    # Valutazione RAG (retrieval + generazione)
    erag_results = erag_mod.eval(
        retrieval_results=retrieval_results_dict,
        expected_outputs=expected_outputs,
        text_generator=t5_generator_for_eval,
        downstream_metric=downstream_metric_func,
        retrieval_metrics=retrieval_metrics
    )

    # Salvataggi
    per_input_file = os.path.join(log_dir, f"per_input_{method}.json")
    save_json_log(erag_results['per_input'], per_input_file, "Risultati per-input")

    aggregated_file = os.path.join(log_dir, f"aggregated_{method}.json")
    save_json_log(erag_results['aggregated'], aggregated_file, "Risultati aggregati")

    return erag_results


def evaluation_e2e(
    expected_outputs,
    retrieval_results_dict,
    t5_generator_for_eval,
    downstream_metric_func,
    test_queries,
    method,
    k_values,
    log_dir="../logs",
):
    # Preparazione container per tutti i k
    all_e2e_scores = {}
    average_e2e_scores = {}

    # Itera su ogni k in k_values e valuta end-to-end limitando i documenti a k
    for k in sorted(set(k_values)):
        print(f"\nEvaluating end-to-end scores for k={k}...")
        # Limita i documenti per query a k
        retrieval_results_topk = _limit_docs_per_query(retrieval_results_dict, k)

        # Generazione end-to-end e punteggi per questo k
        end_to_end_generated = t5_generator_for_eval(retrieval_results_topk)
        e2e_scores_dict = downstream_metric_func(end_to_end_generated, expected_outputs)

        # Garantisce che tutte le query siano presenti
        e2e_scores_dict = {q: e2e_scores_dict.get(q, 0) for q in test_queries}

        # Media
        average_e2e_score = (sum(e2e_scores_dict.values()) / len(e2e_scores_dict)) if e2e_scores_dict else 0.0

        # Salva nel contenitore per questo k
        all_e2e_scores[f"k_{k}"] = e2e_scores_dict
        average_e2e_scores[f"k_{k}"] = average_e2e_score

    # Salva tutto in un unico file di log
    e2e_file = os.path.join(log_dir, f"end_to_end_{method}.json")
    e2e_to_save = {
        "average_scores": average_e2e_scores,
        "per_k_scores": all_e2e_scores,
    }
    save_json_log(e2e_to_save, e2e_file, "Punteggi end-to-end")

    return all_e2e_scores, average_e2e_scores


def get_correlations(
    erag_results,
    retrieval_metrics,
    all_e2e_scores,
    method,
    doc_n,
    log_dir="../logs",
):
    def _select_k_key_for_metric(metric_name, fallback):
        m = re.search(r'(\d+)$', metric_name)
        k = int(m.group(1)) if m else fallback
        return f"k_{k}", k

    correlations = {}

    for metric_name in retrieval_metrics:
        e2e_key, k_used = _select_k_key_for_metric(metric_name, doc_n)
        e2e_scores_dict = all_e2e_scores[e2e_key]

        aligned_erag_scores, aligned_e2e_scores = zip(*[
            (
                erag_results['per_input'].get(q, {}).get(metric_name),
                e2e_scores_dict.get(q)
            )
            for q in e2e_scores_dict.keys()
        ])

        spearman_corr, spearman_p = stats.spearmanr(aligned_erag_scores, aligned_e2e_scores)
        kendall_corr, kendall_p = stats.kendalltau(aligned_erag_scores, aligned_e2e_scores)
        corr_entry = {
            "spearman_corr": spearman_corr,
            "spearman_p": spearman_p,
            "kendall_corr": kendall_corr,
            "kendall_p": kendall_p,
        }

        print(f"\nEvaluated {len(aligned_erag_scores)} pairs.")
        print(f"For metric {metric_name} ({method}, k={k_used}):")
        print(f"  Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.3f})")
        print(f"  Kendall correlation:   {kendall_corr:.3f} (p={kendall_p:.3f})")

        correlations[metric_name] = corr_entry

    corr_file = os.path.join(log_dir, f"correlations_{method}.json")
    save_json_log(correlations, corr_file, "Correlazioni retrieval vs end-to-end")
    return correlations

def define_retrieval_metrics(k_values, metric):
    retrieval_metrics = []
    for k in k_values:
        retrieval_metrics.extend([f'P_{k}', f'success_{k}'])
        if metric != 'f1':
            retrieval_metrics.extend([f'recall_{k}', f'ndcg_cut_{k}', f'map_cut_{k}', f'recip_rank_cut_{k}'])
    # if metric != 'f1':
    #     retrieval_metrics.append(f'recip_rank') # MRR senza cutoff
    return max(k_values), retrieval_metrics


def full_evaluation(args):
    # 0) Preparazione log directory
    LOG_DIR = "../logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1) Definizione metriche
    doc_n, retrieval_metrics = define_retrieval_metrics(args.k_values, args.metric)
    selected_metric_func = METRICS[args.metric]
    print(f"\nUsing evaluation metric: {args.metric.upper()}")

    # 2) Caricamento dataset di test
    print(f"Loading test dataset queries and expected outputs from: {args.test_dataset_path}")
    expected_outputs = load_expected_outputs(args.test_dataset_path)
    test_queries = sorted(list(expected_outputs.keys()))
    print(f"Loaded {len(test_queries)} test queries.") 

    # 3) Retrieval sui dati di test
    print(f"Retrieving {doc_n} documents per query using: {args.method}")
    retriever = None
    if args.method == 'Contriever':
        retriever = DenseRetriever(
            index_path="./index_out_full/ivfpq_opq_contriever.faiss",
            collection_path="../data/collection/wikipedia_passages.jsonl",
            offsets_path="./index_out_full/collection_offsets.u64.bin",
            nprobe=64,
        )
    test_retrieval_results = retrieval_results(test_queries, method=args.method, k=doc_n, retriever=retriever)
    print(f"Documents retrieved.")

    torch.cuda.empty_cache()

    # 4) Caricamento modello T5
    print(f"Loading model from: {args.model_dir}")
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    
    # Create a partial function that has model, tokenizer, device, etc. pre-filled
    t5_generator_for_eval = partial(
        t5_fid_generator,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_input_len=256,
        max_output_len=64,
        num_beams=4
    )

    # 5) Valutazione eRAG
    erag_results = evaluation_erag(
        expected_outputs=expected_outputs,
        retrieval_results_dict=test_retrieval_results,
        t5_generator_for_eval=t5_generator_for_eval,
        downstream_metric_func=selected_metric_func,
        retrieval_metrics=retrieval_metrics,
        method=args.method,
        log_dir=LOG_DIR
    )

    # 6) Valutazione end-to-end per ogni k in k_values
    all_e2e_scores, average_e2e_scores = evaluation_e2e(
        expected_outputs=expected_outputs,
        retrieval_results_dict=test_retrieval_results,
        t5_generator_for_eval=t5_generator_for_eval,
        downstream_metric_func=selected_metric_func,
        test_queries=test_queries,
        method=args.method,
        k_values=args.k_values,
        log_dir=LOG_DIR
    )

    # 7) Correlazioni tra metriche eRAG e punteggi end-to-end
    correlations = get_correlations(
        erag_results=erag_results,
        retrieval_metrics=retrieval_metrics,
        all_e2e_scores=all_e2e_scores,
        method=args.method,
        doc_n=doc_n,
        log_dir=LOG_DIR
    )

    return {
        "retrieval_metrics": retrieval_metrics,
        "erag_results": erag_results,
        "all_e2e_scores": all_e2e_scores,
        "e2e_average_scores": average_e2e_scores,
        "correlations": correlations,
    }


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--model_dir", type=str, default="../models/fid_t5",
                        help="Model directory path")
    parser.add_argument("--k_values", type=int, nargs="+", default=[50],
                        help="List of cutoff values to use for metrics computation. (Highiest will be used as number of retrieved docs)")
    parser.add_argument("--method", type=str, default="BM25", choices=["BM25", "Contriever"],
                        help="Retrieval method to use (BM25 or Contriever). Default is 'BM25'.")
    parser.add_argument("--test_dataset_path", type=str, default="../data/nq-dev-kilt.jsonl",
                        help="Validation file path")
    parser.add_argument("--metric", type=str, default="em", choices=METRICS.keys(),
                        help=f"Evaluation metric to use. Choices: {list(METRICS.keys())}. Default is 'em' (exact_match).")
    args = parser.parse_args()
    full_evaluation(args)