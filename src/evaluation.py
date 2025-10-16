import json
import string
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


def model_loading(args):
    
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

    print(f"Retrieving documents using: {args.method}")
    retrieve_results = retrieval_results(test_queries, method=args.method, k=args.doc_n, retriever=retriever)
    print(f"Documents retrieved.")

    torch.cuda.empty_cache()
    
    model_path = args.model_path
    max_input_len = 256
    max_output_len = 64
    num_beams_eval = 4

    print(f"Loading model from: {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
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

 # Create a dictionary to map metric names (strings) to their corresponding functions
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
        print(f"{description} salvato in {file_path}.")
    else:
        print(f"Salvato in {file_path}.")


def evaluation(args):
    test_expected_outputs, test_retrieval_results, t5_generator_for_eval, test_queries_set = model_loading(args)
    selected_metric_func = METRICS[args.metric]
    print(f"\nUsing evaluation metric: {args.metric.upper()}")

    # Use a sorted list of queries for consistent order in evaluations
    test_queries_list = sorted(list(test_queries_set))

    LOG_DIR = "../logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    doc_n = args.doc_n
    k_values = args.k_values
    method = args.method

    print(f"\n--- Processing {method} with doc_n = {doc_n} ---")

    # Define retrieval metrics based on k and the chosen downstream metric
    retrieval_metrics = []
    for k in k_values:
        if k > doc_n:
            continue
        retrieval_metrics.extend([f'P_{k}', f'success_{k}'])
        if args.metric != 'f1':
            retrieval_metrics.extend([f'recall_{k}', f'ndcg_cut_{k}'])

    if args.metric != 'f1':
        retrieval_metrics.extend(['map', 'recip_rank'])

    # Evaluate retrieval and generation
    erag_results = erag.eval(
        retrieval_results=test_retrieval_results,
        expected_outputs=test_expected_outputs,
        text_generator=t5_generator_for_eval,
        downstream_metric=selected_metric_func,
        retrieval_metrics=retrieval_metrics
    )

    # Salvataggi semplificati con funzione helper
    per_input_file = os.path.join(LOG_DIR, f"per_input_{method}_doc_n{doc_n}.json")
    save_json_log(erag_results['per_input'], per_input_file, "Risultati per-input")

    aggregated_file = os.path.join(LOG_DIR, f"aggregated_{method}_doc_n{doc_n}.json")
    save_json_log(erag_results['aggregated'], aggregated_file, "Risultati aggregati")

    # Generate end-to-end responses e calcolo punteggi
    end_to_end_generated = t5_generator_for_eval(test_retrieval_results)
    e2e_scores_dict = selected_metric_func(end_to_end_generated, test_expected_outputs)
    # Garantisce che tutte le query siano presenti
    e2e_scores_dict = {q: e2e_scores_dict.get(q, 0) for q in test_queries_list}

    # Media e2e
    average_e2e_score = (sum(e2e_scores_dict.values()) / len(e2e_scores_dict)) if e2e_scores_dict else 0.0

    # Salva media e punteggi e2e nello stesso file (media in cima)
    e2e_file = os.path.join(LOG_DIR, f"end_to_end_{method}_doc_n{doc_n}.json")
    e2e_to_save = {"average_score": average_e2e_score}
    e2e_to_save.update(e2e_scores_dict)
    save_json_log(e2e_to_save, e2e_file, "Punteggi end-to-end")

    # Calcolo e salvataggio correlazioni
    correlations = {}
    for metric_name in retrieval_metrics:
        aligned_erag_scores = []
        aligned_e2e_scores = []
        num_queries_with_metric = 0

        for query_id in test_queries_list:
            query_result_dict = erag_results['per_input'].get(query_id, {})
            erag_score = query_result_dict.get(metric_name)
            if erag_score is not None:
                aligned_erag_scores.append(erag_score)
                aligned_e2e_scores.append(e2e_scores_dict[query_id])
                num_queries_with_metric += 1

        corr_entry = {
            "num_pairs": num_queries_with_metric,
            "spearman_corr": None,
            "spearman_p": None,
            "kendall_corr": None,
            "kendall_p": None,
        }

        spearman_corr, spearman_p = stats.spearmanr(aligned_erag_scores, aligned_e2e_scores)
        kendall_corr, kendall_p = stats.kendalltau(aligned_erag_scores, aligned_e2e_scores)
        corr_entry.update({
            "spearman_corr": float(spearman_corr),
            "spearman_p": float(spearman_p),
            "kendall_corr": float(kendall_corr),
            "kendall_p": float(kendall_p),
        })

        print(f"\nFor metric {metric_name} ({method}, doc_n={doc_n}):")
        print(f"  Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.3f})")
        print(f"  Kendall correlation:   {kendall_corr:.3f} (p={kendall_p:.3f})")

        correlations[metric_name] = corr_entry

    corr_file = os.path.join(LOG_DIR, f"correlations_{method}_doc_n{doc_n}.json")
    save_json_log(correlations, corr_file, "Correlazioni retrieval vs end-to-end")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--model_dir", type=str, required=True, default="../models/fid_t5",
                        help="Model directory path")
    parser.add_argument("--k_values", type=int, nargs="+", required=True, default=[50],
                        help="List of cut values to use for metrics computation.")
    parser.add_argument("--method", type=str, default="BM25", choices=["BM25", "Contriever"],
                        help="Retrieval method to use (BM25 or Contriever). Default is 'BM25'.")
    parser.add_argument("--doc_n", type=int, default=50,
                        help="Number of retrieved documents to use. Default is 50.")
    parser.add_argument("--test_dataset_path", type=str, required=True, default="../data/nq-dev-kilt.jsonl",
                        help="Validation file path")
    parser.add_argument("--metric",
                        type=str,
                        default="em",
                        choices=METRICS.keys(),
                        help=f"Evaluation metric to use. Choices: {list(METRICS.keys())}. Default is 'em' (exact_match).")
    args = parser.parse_args()
    evaluation(args)