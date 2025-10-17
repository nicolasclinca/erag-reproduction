import json
import string
import re
import torch
import erag
import os
import scipy.stats as stats
from functools import partial
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_loader import retrieval_results, load_expected_outputs
from collections import Counter
import argparse
from fid_t5 import t5_fid_generator
from src.contriever_retriever import DenseRetriever


def model_loading(args, doc_n=50):
    
    print(f"Loading test dataset queries and expected outputs from: {args.test_dataset_path}")
    expected_outputs = load_expected_outputs(args.test_dataset_path)
    test_queries = list(expected_outputs.keys())
    print(f"Loaded {len(test_queries)} test queries.")      

    retriever = None
    if args.method == 'Contriever':
        index_path = "./index_out_full/ivfpq_opq_contriever.faiss"
        collection_path = "../data/collection/wikipedia_passages.jsonl"
        offsets_path = "./index_out_full/collection_offsets.u64.bin"
        nprobe = 64

        retriever = DenseRetriever(
            index_path=index_path,
            collection_path=collection_path,
            offsets_path=offsets_path,
            nprobe=nprobe,
        )

    print(f"Retrieving {doc_n} documents per query using: {args.method}")
    retrieve_results = retrieval_results(test_queries, method=args.method, k=doc_n, retriever=retriever)
    print(f"Documents retrieved.")

    torch.cuda.empty_cache()
    
    model_dir = args.model_dir
    max_input_len = 256
    max_output_len = 64
    num_beams_eval = 4

    print(f"Loading model from: {model_dir}")
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
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
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        num_beams=num_beams_eval
    )
    
    return expected_outputs, retrieve_results, t5_generator_for_eval, test_queries

    
def normalize_answer(s):
    """Converts text to lowercase, removes punctuation and extra spaces."""
    return ' '.join(''.join(ch for ch in s.lower() if ch not in string.punctuation).split())

def exact_match_metric(generated_outputs, expected_outputs):
    """Computes if the generated text (normalized) exactly matches one of the gold answers (normalized).
    Returns a dict {query: score}, where score is 1 or 0."""
    return {query: 1 if any(normalize_answer(gen) == normalize_answer(gold) for gold in expected_outputs.get(query, [])) else 0 for query, gen in generated_outputs.items()}

def _f1_score(prediction, ground_truth):
    """Helper function to compute F1 score for a single prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_metric(generated_outputs, expected_outputs):
    """
    Computes the F1 score for each query.
    For each query, it takes the maximum F1 score over all possible gold answers.
    Returns a dict {query: f1_score}.
    """
    f1_scores = {}
    for query, gen_answer in generated_outputs.items():
        gold_answers = expected_outputs.get(query, [])
        if not gold_answers:
            f1_scores[query] = 0.0
            continue
        max_f1 = max(_f1_score(gen_answer, gold) for gold in gold_answers)
        f1_scores[query] = max_f1
    return f1_scores

METRICS = {
    "em": exact_match_metric,
    "f1": f1_metric,
    "accuracy": exact_match_metric # In our case exact match and accuracy coincide
}

def save_json_log(data, file_path, description=None):
    """
    Salva un dizionario in JSON e stampa un messaggio di conferma.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if description:
        print(f"{description} saved in {file_path}.")
    else:
        print(f"Saved in {file_path}.")

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

    print(f"\nEvaluating using eRAG...")

    # Valutazione RAG (retrieval + generazione)
    erag_results = erag.eval(
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
    test_queries_list,
    method,
    k_values,
    log_dir="../logs",
):
    # Preparazione container per tutti i k
    all_e2e_scores = {}
    average_e2e_scores = {}

    # Itera su ogni k in k_values e valuta end-to-end limitando i documenti a k
    for k in sorted(set(k_values)):
        print(f"\nEvaluating end-to-end for k={k}...")
        # Limita i documenti per query a k
        retrieval_results_topk = _limit_docs_per_query(retrieval_results_dict, k)

        # Generazione end-to-end e punteggi per questo k
        end_to_end_generated = t5_generator_for_eval(retrieval_results_topk)
        e2e_scores_dict = downstream_metric_func(end_to_end_generated, expected_outputs)

        # Garantisce che tutte le query siano presenti
        e2e_scores_dict = {q: e2e_scores_dict.get(q, 0) for q in test_queries_list}

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
    test_queries_list,
    method,
    doc_n,
    log_dir="../logs",
):
    def _select_k_key_for_metric(metric_name, fallback):
        """
        Dato il nome della metrica (es. 'P_10', 'ndcg_cut_5'),
        prova a estrarre k. Se non presente, usa fallback.
        """
        m = re.search(r'(\d+)$', metric_name)
        if m:
            k = int(m.group(1))
        else:
            k = fallback

        key = f"k_{k}"
        return key, k

    correlations = {}

    for metric_name in retrieval_metrics:
        e2e_key, k_used = _select_k_key_for_metric(metric_name, doc_n)
        e2e_scores_dict = all_e2e_scores[e2e_key]

        aligned_erag_scores = []
        aligned_e2e_scores = []

        for query_id in test_queries_list:
            query_result_dict = erag_results['per_input'].get(query_id, {})
            erag_score = query_result_dict.get(metric_name)
            if erag_score is not None:
                aligned_erag_scores.append(erag_score)
                aligned_e2e_scores.append(e2e_scores_dict.get(query_id, 0))

        corr_entry = {
            "num_pairs": len(aligned_erag_scores),
            "spearman_corr": None,
            "spearman_p": None,
            "kendall_corr": None,
            "kendall_p": None,
            "e2e_k_used": k_used,
        }

        # Calcolo correlazioni solo se ci sono abbastanza coppie e non sono costanti
        can_corr = (
            len(aligned_erag_scores) >= 2 and
            len(set(aligned_erag_scores)) > 1 and
            len(set(aligned_e2e_scores)) > 1
        )

        if can_corr:
            spearman_corr, spearman_p = stats.spearmanr(aligned_erag_scores, aligned_e2e_scores)
            kendall_corr, kendall_p = stats.kendalltau(aligned_erag_scores, aligned_e2e_scores)
            corr_entry.update({
                "spearman_corr": float(spearman_corr),
                "spearman_p": float(spearman_p),
                "kendall_corr": float(kendall_corr),
                "kendall_p": float(kendall_p),
            })

            print(f"\nFor metric {metric_name} ({method}, k={k_used}):")
            print(f"  Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.3f})")
            print(f"  Kendall correlation:   {kendall_corr:.3f} (p={kendall_p:.3f})")
        else:
            print(f"\nFor metric {metric_name} ({method}, k={k_used}):")
            print("  Spearman correlation: N/A (dati insufficienti o costanti)")
            print("  Kendall correlation:  N/A (dati insufficienti o costanti)")

        correlations[metric_name] = corr_entry

    corr_file = os.path.join(log_dir, f"correlations_{method}.json")
    save_json_log(correlations, corr_file, "Correlazioni retrieval vs end-to-end")
    return correlations


def define_retrieval_metrics(args):
    k_values = args.k_values
    retrieval_metrics = []
    for k in k_values:
        retrieval_metrics.extend([f'P_{k}', f'success_{k}'])
        if args.metric != 'f1':
            retrieval_metrics.extend([f'recall_{k}', f'ndcg_cut_{k}', f'map_{k}', f'recip_rank_{k}'])
    return max(k_values), retrieval_metrics


def full_evaluation(args):
    LOG_DIR = "../logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Definizione metriche
    doc_n, retrieval_metrics = define_retrieval_metrics(args)
    selected_metric_func = METRICS[args.metric]
    print(f"\nUsing evaluation metric: {args.metric.upper()}")

    # 1) Caricamento modello e dati
    test_expected_outputs, test_retrieval_results, t5_generator_for_eval, test_queries_set = model_loading(args, doc_n=doc_n)
    test_queries_list = sorted(list(test_queries_set))

    # 2) Valutazione eRAG
    erag_results = evaluation_erag(
        expected_outputs=test_expected_outputs,
        retrieval_results_dict=test_retrieval_results,
        t5_generator_for_eval=t5_generator_for_eval,
        downstream_metric_func=selected_metric_func,
        retrieval_metrics=retrieval_metrics,
        method=args.method,
        log_dir=LOG_DIR
    )

    # 3) Valutazione end-to-end per ogni k in k_values
    all_e2e_scores, average_e2e_scores = evaluation_e2e(
        expected_outputs=test_expected_outputs,
        retrieval_results_dict=test_retrieval_results,
        t5_generator_for_eval=t5_generator_for_eval,
        downstream_metric_func=selected_metric_func,
        test_queries_list=test_queries_list,
        method=args.method,
        k_values=args.k_values,
        log_dir=LOG_DIR
    )

    # 4) Correlazioni tra metriche eRAG e punteggi end-to-end
    correlations = get_correlations(
        erag_results=erag_results,
        retrieval_metrics=retrieval_metrics,
        all_e2e_scores=all_e2e_scores,
        test_queries_list=test_queries_list,
        method=args.method,
        doc_n=doc_n,
        log_dir=LOG_DIR
    )

    return {
        "erag_results": erag_results,
        "retrieval_metrics": retrieval_metrics,
        "all_e2e_scores": all_e2e_scores,
        "e2e_average_scores": average_e2e_scores,
        "correlations": correlations,
    }


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--model_dir", type=str, required=True, default="../models/fid_t5",
                        help="Model directory path")
    parser.add_argument("--k_values", type=int, nargs="+", required=True, default=[50],
                        help="List of cutoff values to use for metrics computation.")
    parser.add_argument("--method", type=str, default="BM25", choices=["BM25", "Contriever"],
                        help="Retrieval method to use (BM25 or Contriever). Default is 'BM25'.")
    parser.add_argument("--test_dataset_path", type=str, required=True, default="../data/nq-dev-kilt.jsonl",
                        help="Validation file path")
    parser.add_argument("--metric",
                        type=str,
                        default="em",
                        choices=METRICS.keys(),
                        help=f"Evaluation metric to use. Choices: {list(METRICS.keys())}. Default is 'em' (exact_match).")
    args = parser.parse_args()
    full_evaluation(args)