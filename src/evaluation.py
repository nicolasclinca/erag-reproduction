import json
import string
import time
import torch
import erag
import os
import pickle
import scipy.stats as stats
from functools import partial
from transformers.modeling_outputs import BaseModelOutput
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_loader import retrieval_results
from collections import Counter
import argparse


def T5_text_generator(
    queries_and_documents: dict,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    device: torch.device,
    max_input_len: int = 512,
    max_output_len: int = 128,
    num_beams: int = 4,
    **generate_kwargs):
    """
    Generates answers for multiple queries using a fine-tuned T5 FiD model.

    Args:
        queries_and_documents: Dictionary where keys are query strings and
                               values are lists of retrieved document strings.
        model: The loaded fine-tuned T5ForConditionalGeneration model (on device).
        tokenizer: The loaded corresponding T5Tokenizer.
        device: The torch.device where the model is located.
        max_input_len: Max sequence length for each (query + doc) input.
        max_output_len: Max sequence length for the generated answer.
        num_beams: Number of beams for beam search generation.
        **generate_kwargs: Additional keyword arguments passed to model.generate().

    Returns:
        Dictionary where keys are the input query strings and values are the
        corresponding generated answer strings (or an error message if no docs provided).
    """
    model.eval()
    results = {}
    pad_token_id = tokenizer.pad_token_id

    print(f"Generating answers for {len(queries_and_documents)} queries...")
    start_time_total = time.time()

    # Iterate through each query and its associated documents
    for i, (query, retrieved_docs) in enumerate(queries_and_documents.items()):
        start_time_query = time.time()
        print(f"  Processing query {i+1}/{len(queries_and_documents)}: \"{query[:50]}...\"")

        if not retrieved_docs:
            print(f"    Warning: No documents found for query {i+1}. Skipping.")
            results[query] = "Error: No documents provided for this query."
            continue

        all_input_ids = []
        all_attention_masks = []

        # --- Core FiD Generation Logic (applied per query) ---

        # 1. Preprocess and Tokenize each document for the CURRENT query
        for doc in retrieved_docs:
            input_text = f"question: {query} context: {doc}"
            encoding = tokenizer(
                input_text,
                truncation=True,
                max_length=max_input_len,
                padding="max_length",
                return_attention_mask=True,
                add_special_tokens=True
            )
            all_input_ids.append(torch.tensor(encoding['input_ids']))
            all_attention_masks.append(torch.tensor(encoding['attention_mask']))

        # 2. Stack inputs and move to device for CURRENT query
        input_ids_stacked = torch.stack(all_input_ids).to(device)
        attention_mask_stacked = torch.stack(all_attention_masks).to(device)
        num_docs, seq_len = input_ids_stacked.shape

        # Use no_grad context for efficiency during inference
        with torch.no_grad():
            # 3. Encoder Pass for CURRENT query
            raw_encoder_outputs = model.encoder(
                input_ids=input_ids_stacked,
                attention_mask=attention_mask_stacked,
                return_dict=True
            )

            # 4. Reshape Encoder Outputs for Generate
            encoder_hidden_states_reshaped = raw_encoder_outputs.last_hidden_state.view(
                1, num_docs * seq_len, raw_encoder_outputs.last_hidden_state.size(-1)
            )
            encoder_outputs_for_generate = BaseModelOutput(
                last_hidden_state=encoder_hidden_states_reshaped
            )

            # 5. Prepare Cross-Attention Mask
            cross_attention_mask_reshaped = attention_mask_stacked.view(1, num_docs * seq_len)

            # 6. Generation (Decoder) using model.generate() for CURRENT query
            generated_ids = model.generate(
                encoder_outputs=encoder_outputs_for_generate, # Reshaped encoder outputs
                attention_mask=cross_attention_mask_reshaped, # Mask for cross-attention
                max_length=max_output_len,
                num_beams=num_beams,
                early_stopping=True,
                **generate_kwargs
            )

        # 7. Decode the generated token IDs back to text
        generated_text = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )

        # Store the generated answer in the results dictionary
        results[query] = generated_text.strip()
        end_time_query = time.time()
        # print(f"    Generated answer in {end_time_query - start_time_query:.2f} seconds.")

    end_time_total = time.time()
    print(f"Finished generating all answers in {end_time_total - start_time_total:.2f} seconds.")

    # Return the dictionary containing {query: answer} pairs
    return results





#def model_loading(expected_outputs, retrieve_results, test_data):
def model_loading(args):
    # Load the test data
    filename = args.filename
    expected_outputs, retrieve_results = retrieval_results(filename=filename)
    model_path = "../models/finetuned_t5_model_fid"
    max_input_len = 512
    max_output_len = 128
    num_beams_eval = 4

    # Load the fine-tuned model and tokenizer
    print(f"Loading model from: {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}")
    
    # Create a partial function that has model, tokenizer, device, etc. pre-filled
    t5_generator_for_eval = partial(
        T5_text_generator,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        num_beams=num_beams_eval
    )
    
    with open("../data/augmented_dev.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    #Loading the Test set queries
    test_queries = set(item['query'] for item in test_data)
    print(f"\nExtracted {len(test_queries)} unique queries for the test set.")

    # 2. Create the test split dictionaries using the test queries
    test_expected_outputs = {
        query: answers
        for query, answers in expected_outputs.items()
        if query in test_queries
    }

    test_retrieval_results = {
        query: docs
        for query, docs in retrieve_results.items()
        if query in test_queries
    }

    print("\nVerification:")
    print(f"Size of test_expected_outputs: {len(test_expected_outputs)} (should match test_data size)")
    print(f"Size of test_retrieval_results: {len(test_retrieval_results)} (should match test_data size)")
    
    return test_expected_outputs, test_retrieval_results, t5_generator_for_eval, test_queries

    
# Defining the evaluation function for Erag
# Define the Exact Match evaluation function
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

    # if empty 
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

def accuracy_metric(generated_outputs, expected_outputs):
    """
    Computes if the generated text (normalized) exactly matches one of the gold answers (normalized).
    This is equivalent to the Exact Match metric and returns a score of 1 for a match and 0 otherwise.
    Returns a dict {query: score}, where score is 1 or 0.
    """
    return {query: 1 if any(normalize_answer(gen) == normalize_answer(gold) for gold in expected_outputs.get(query, [])) else 0 for query, gen in generated_outputs.items()}

 # Create a dictionary to map metric names (strings) to their corresponding functions
METRICS = {
    "em": exact_match_metric,
    "f1": f1_metric,
    "accuracy": accuracy_metric
}

# Evaluation loop
def evaluation(args):
    test_expected_outputs, test_retrieval_results, t5_generator_for_eval, test_queries_set = model_loading(args)
    # Get the selected metric function from the dictionary based on the command-line argument
    selected_metric_func = METRICS[args.metric]
    print(f"\nUsing evaluation metric: {args.metric.upper()}")
    # Use a sorted list of queries for consistent order in evaluations
    test_queries_list = sorted(list(test_queries_set))
    # Create the log directory if it doesn't exist
    LOG_DIR = "../logs"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Define checkpoint file
    #CHECKPOINT_FILE = "experiment_checkpoint.pkl"
    CHECKPOINT_FILE = "../logs/experiment_checkpoint.pkl"

    # Load checkpoint if it exists, otherwise initialize an empty dictionary
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            checkpoint = pickle.load(f)
        print("Checkpoint loaded.")
    else:
        checkpoint = {}
        print("No checkpoint found, starting from scratch.")



    # Define values for K (number of retrieved documents) and retrieval methods
    k_values = [args.k_values]
    retriever_methods = ['BM25'] # , 'dense'

    # Load existing correlations from checkpoint
    correlations = {method: checkpoint.get(method, {}) for method in retriever_methods}

    # Iterate over retrieval methods and values of K
    for method in retriever_methods:
        for k in k_values:
            if k in checkpoint.get(method, {}):
                print(f"Skipping {method} with K={k} (already processed).")
                continue

            print(f"\n--- Processing {method} with K = {k} ---")
            try:
                # Define retrieval metrics based on k
                #retrieval_metrics = {'P', 'success', 'recall', 'map', 'ndcg', 'recip_rank'}
                retrieval_metrics = [
                    f'P_{k}',
                    f'success_{k}',
                    f'recall_{k}',
                    'map',
                    f'ndcg_cut_{k}',
                    'recip_rank'
                ]
                # Initialize ERAG
                #erag = ERAG

                # Evaluate retrieval and generation
                erag_results = erag.eval(
                    retrieval_results=test_retrieval_results,
                    expected_outputs=test_expected_outputs,
                    text_generator=t5_generator_for_eval,
                    downstream_metric=selected_metric_func,
                    retrieval_metrics=retrieval_metrics
                )

                # Save per-query results
                per_input_file = os.path.join(LOG_DIR, f"per_input_{method}_K{k}.json")
                with open(per_input_file, "w", encoding="utf-8") as f:
                    json.dump(erag_results['per_input'], f, ensure_ascii=False, indent=2)
                print(f"Saved per-input results in {per_input_file}.")

                # Save aggregated results
                aggregated_file = os.path.join(LOG_DIR, f"aggregated_{method}_K{k}.json")
                with open(aggregated_file, "w", encoding="utf-8") as f:
                    json.dump(erag_results['aggregated'], f, ensure_ascii=False, indent=2)
                print(f"Saved aggregated results in {aggregated_file}.")

                # Generate end-to-end responses
                end_to_end_generated = t5_generator_for_eval(test_retrieval_results)
                # Ensure e2e_scores_dict covers all queries in test_queries_list, defaulting to 0 if a query somehow wasn't processed
                e2e_scores_dict = {q: score for q, score in selected_metric_func(end_to_end_generated, test_expected_outputs).items()}

                # Save end-to-end scores
                e2e_file = os.path.join(LOG_DIR, f"end_to_end_{method}_K{k}.json")
                with open(e2e_file, "w", encoding="utf-8") as f:
                    # Save scores for all test queries, ensuring consistent structure
                    scores_to_save = {q: e2e_scores_dict.get(q, 0) for q in test_queries_list}
                    json.dump(scores_to_save, f, ensure_ascii=False, indent=2)
                print(f"Saved end-to-end scores in {e2e_file}.")

                # Compute the average end-to-end score
                average_e2e_score = sum(e2e_scores_dict.values()) / len(e2e_scores_dict)
                
                # Save the aggregated average in a separate file
                aggregated_e2e_file = os.path.join(LOG_DIR, f"aggregated_end_to_end_{method}_K{k}.json")
                with open(aggregated_e2e_file, "w", encoding="utf-8") as f:
                    json.dump({"average_score": average_e2e_score}, f, ensure_ascii=False, indent=2)
                print(f"Saved aggregated end-to-end score in {aggregated_e2e_file}.")
                
                # Compute correlation between retrieval and end-to-end scores
                # This list will be aligned with test_queries_list
                end_to_end_scores_list = [e2e_scores_dict.get(query, 0) for query in test_queries_list]

                local_corr = {}
                for metric_name in retrieval_metrics: # Iterate using the base names
                   
                    aligned_erag_scores = []
                    aligned_e2e_scores = []
                    num_queries_with_metric = 0

                    for idx, query_id in enumerate(test_queries_list):
                        query_result_dict = erag_results['per_input'].get(query_id, {})
                        erag_score = query_result_dict.get(metric_name)

                        if erag_score is not None:
                            aligned_erag_scores.append(erag_score)
                            aligned_e2e_scores.append(end_to_end_scores_list[idx])
                            num_queries_with_metric += 1
                        # else: erag_score is None, so we skip this query for this metric's correlation

                    if num_queries_with_metric < 2:
                        print(f"  Skipping correlation for {metric_name} for K={k}: fewer than 2 queries with this metric ({num_queries_with_metric} found).")
                        spearman_corr, spearman_p = float('nan'), float('nan')
                        kendall_corr, kendall_p = float('nan'), float('nan')
                    elif len(set(aligned_erag_scores)) < 2 or len(set(aligned_e2e_scores)) < 2:
                        print(f"  Skipping correlation for {metric_name} for K={k}: insufficient variance in scores ({num_queries_with_metric} pairs).")
                        spearman_corr, spearman_p = float('nan'), float('nan')
                        kendall_corr, kendall_p = float('nan'), float('nan')
                    else:
                        spearman_corr, spearman_p = stats.spearmanr(aligned_erag_scores, aligned_e2e_scores)
                        kendall_corr, kendall_p = stats.kendalltau(aligned_erag_scores, aligned_e2e_scores)

                    local_corr[metric_name] = { # Store with base_metric_name for consistency
                        'spearman': spearman_corr,
                        'kendall': kendall_corr,
                        'num_queries_correlated': num_queries_with_metric
                    }
                    print(f"\nFor metric {metric_name} ({method}, K={k}):")
                    print(f"  Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.3f})")
                    print(f"  Kendall correlation:   {kendall_corr:.3f} (p={kendall_p:.3f})")

                # Update checkpoint
                if method not in checkpoint:
                    checkpoint[method] = {}
                checkpoint[method][k] = local_corr
                # Update the correlations dictionary for the final printout
                correlations[method][k] = local_corr
                with open(CHECKPOINT_FILE, "wb") as f:
                    pickle.dump(checkpoint, f)
                print(f"Checkpoint updated for {method} with K={k}.")

            except Exception as e:
                print(f"Error for {method} with K={k}: {e}") # This prints str(e)
                print(f"  Exception Type: {type(e)}")
                print(f"  Exception Repr: {repr(e)}")
                print(f"  Exception Args: {e.args}")
                import traceback
                print("--- Traceback ---")
                traceback.print_exc()
                print("--- End Traceback ---")
                time.sleep(10)
                continue

    print("\n--- Final correlation summary ---")
    print(correlations)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--k_values", type=int, default=50,
                        help="Number of retireved document. Default is 50")
    parser.add_argument("--filename", type=str, required=True, default="../data/nq-dev-kilt.jsonl",
                        help="Validation file name")
    parser.add_argument("--metric",
                        type=str,
                        default="em",
                        choices=METRICS.keys(),
                        help=f"Evaluation metric to use. Choices: {list(METRICS.keys())}. Default is 'em' (exact_match).")
    args = parser.parse_args()
    evaluation(args)