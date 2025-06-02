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
def model_loading():
    test_data = "../data/nq-dev-kilt.jsonl"
    expected_outputs, retrieve_results = retrieval_results(filename=test_data)
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


# Evaluation loop
def evaluation(args):
    test_expected_outputs, test_retrieval_results, t5_generator_for_eval, test_queries = model_loading()
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
    k_values = args.k_values
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
                retrieval_metrics = {f'P_{k}', f'success_{k}', f'recall_{k}', f'map_{k}', f'ndcg_{k}', f'recip_rank_{k}'}

                # Initialize ERAG
                #erag = ERAG

                # Evaluate retrieval and generation
                erag_results = erag.eval(
                    retrieval_results=test_retrieval_results,
                    expected_outputs=test_expected_outputs,
                    text_generator=t5_generator_for_eval,
                    downstream_metric=exact_match_metric,
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
                e2e_scores_dict = exact_match_metric(end_to_end_generated, test_expected_outputs)

                # Save end-to-end scores
                e2e_file = os.path.join(LOG_DIR, f"end_to_end_{method}_K{k}.json")
                with open(e2e_file, "w", encoding="utf-8") as f:
                    json.dump(e2e_scores_dict, f, ensure_ascii=False, indent=2)
                print(f"Saved end-to-end scores in {e2e_file}.")

                # Compute correlation between retrieval and end-to-end scores
                end_to_end_scores = [e2e_scores_dict.get(query, 0) for query in test_queries]

                local_corr = {}
                for metric_key in retrieval_metrics:
                    eRAG_scores = [erag_results['per_input'][query].get(metric_key, None) for query in test_queries]
                    eRAG_scores = [score for score in eRAG_scores if score is not None]

                    if len(eRAG_scores) != len(end_to_end_scores):
                        print(f"Warning: dimension mismatch for {method}, metric {metric_key}, K={k}")

                    # Compute correlation if variance exists
                    spearman_corr, spearman_p = stats.spearmanr(eRAG_scores, end_to_end_scores)
                    kendall_corr, kendall_p = stats.kendalltau(eRAG_scores, end_to_end_scores)
                    local_corr[metric_key] = {
                        'spearman': spearman_corr,
                        'kendall': kendall_corr,
                        'num_queries': len(eRAG_scores)
                    }
                    print(f"\nFor metric {metric_key} ({method}, K={k}):")
                    print(f"  Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.3f})")
                    print(f"  Kendall correlation:   {kendall_corr:.3f} (p={kendall_p:.3f})")

                # Update checkpoint
                checkpoint[method][k] = local_corr
                with open(CHECKPOINT_FILE, "wb") as f:
                    pickle.dump(checkpoint, f)
                print(f"Checkpoint updated for {method} with K={k}.")

            except Exception as e:
                print(f"Error for {method} with K={k}: {e}")
                time.sleep(10)
                continue

    print("\n--- Final correlation summary ---")
    print(correlations)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--k_values", type=int, default=50,
                        help="Number of retireved document. Default is 50")
    args = parser.parse_args()
    evaluation(args)