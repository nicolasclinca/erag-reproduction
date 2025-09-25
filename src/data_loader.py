import json
import torch
from torch.utils.data import Dataset
from retrieval_models import retrieve_documents
import argparse
import os
from typing import Dict, List, Tuple
from contriever_retrieval import DenseRetriever


"""
Uso CLI:
    python data_loader.py --datasets ../data/nq-train-kilt.jsonl ../data/nq-dev-kilt.jsonl ../data/fever-train-kilt.jsonl ../data/fever-dev-kilt.jsonl ../data/hotpotqa-train-kilt.jsonl ../data/hotpotqa-dev-kilt.jsonl ../data/triviaqa-train-kilt.jsonl ../data/triviaqa-dev-kilt.jsonl ../data/wow-train-kilt.jsonl ../data/wow-dev-kilt.jsonl --method Contriever
"""

# Load all expected outputs from the KILT NQ dev file
def load_all_nq_expected_outputs(filename):
    """Loads all queries and their gold answers from the KILT NQ dev file.
    Returns a dict {query: [gold_answer1, gold_answer2, ...]}."""
    expected = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            query = record["input"].strip()
            golds = [out["answer"].strip() for out in record.get("output", []) if "answer" in out]
            if golds:
                expected[query] = golds
    return expected


def retrieval_results(queries, method='BM25', k=50):
    if method == 'BM25':
        # For each query, retrieval_results[query] = [doc1, doc2, ..., doc50]
        retrieve_res = {query: retrieve_documents(query, method=method, k=k) for query in queries}
    elif method == 'Contriever':
        # Percorsi hardcoded (da gestire meglio con variabili d'ambiente o argomenti)
        index_path = "./index_out_full/ivfpq_opq_contriever.faiss"
        collection_path = "../data/collection/wikipedia_passages.jsonl"
        offsets_path = "./index_out_full/collection_offsets.u64.bin"
        batch_size = 256
        nprobe = 64

        # if not index_path or not collection_path:
        #     raise ValueError(
        #         "Imposta le variabili d'ambiente DENSE_INDEX_PATH e DENSE_COLLECTION_PATH "
        #         "(opzionale: DENSE_OFFSETS_PATH) prima di chiamare retrieval_results()."
        #     )

        # Istanzia il retriever (usa offsets per accesso random veloce se disponibile)
        retr = DenseRetriever(
            index_path=index_path,
            collection_path=collection_path,
            offsets_path=offsets_path,
            nprobe=nprobe,
            in_memory=False,  # True only for mini-run; full-scale: False
        )

        # Batch inference
        retrieve_res: Dict[str, List[str]] = {}
        for i in range(0, len(queries), batch_size):
            batch_q = queries[i : i + batch_size]
            results = retr.batch_dense_retrieve(batch_q, k=k, return_cosine=False)
            for q, r in zip(batch_q, results):
                retrieve_res[q] = r["documents"]
    return retrieve_res


# Add retrieved documents to the dataset
def augment_with_retrieved_documents(nq_dataset, retrieval_results):
    augmented_data = []
    for query, gold_answers in nq_dataset.items():
        # Retrieve corresponding documents for the query
        retrieved_docs = retrieval_results.get(query, [])
        # Target answers (since it's a multi-answer task, we will just take the first answer)
        target_text = gold_answers[0]
        augmented_data.append({"query": query, "retrieved_docs": retrieved_docs, "gold_answer": target_text})
    return augmented_data


def augmented_dataset(args):
    """
    Process a list of datasets (args.datasets). For each path:
      - call retrieval_results(filename=path, method=args.method)
      - call augment_with_retrieved_documents(...)
      - save the result in <same_dir>/<basename>-augmented.json
    """

    method = args.method
    datasets = args.datasets or []

    if not datasets:
        print("No dataset provided in --datasets.")
        return

    for dataset_path in datasets:
        if not dataset_path:
            continue

        if not os.path.exists(dataset_path):
            print(f"File not found: '{dataset_path}'. Skipped.")
            continue

        try:
            # Recupera i risultati di retrieval e costruisce l'augmented dataset
            expected_outputs = load_all_nq_expected_outputs(filename=dataset_path)
            queries = list(expected_outputs.keys())
            print(f"Loaded {len(queries)} queries")
            retrieve_res = retrieval_results(queries=queries, method=method)
            augmented = augment_with_retrieved_documents(expected_outputs, retrieve_res)

            # Costruisce il percorso di output
            dirn = os.path.dirname(dataset_path) or "."
            base = os.path.splitext(os.path.basename(dataset_path))[0]
            out_path = os.path.join(dirn, f"{base}-augmented.json")

            # Salva in JSON (utf-8)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(augmented, f, indent=4)

            print(f"Saved augmented dataset for '{dataset_path}' -> '{out_path}' ({len(augmented)} samples).")

        except Exception as e:
            # Non interrompiamo l'elaborazione degli altri dataset: segnaliamo l'errore e continuiamo
            print(f"Error during the processing of '{dataset_path}': {e}")


# Custom Dataset class
class QA_Dataset_FiD(Dataset):
    def __init__(self, augmented_data, tokenizer, max_input_length=512, max_target_length=128):
        self.data = augmented_data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        gold_answer = item["gold_answer"]
        retrieved_docs = item["retrieved_docs"]

        input_encodings = []
        # retrieved_docs = retrieved_docs[:50]

        for doc in retrieved_docs:
            input_text = f"question: {query} context: {doc}"
            input_encoding = self.tokenizer(input_text, truncation=True, max_length=self.max_input_length)
            input_encodings.append(input_encoding)

        target_text = gold_answer
        target_encoding = self.tokenizer(target_text, truncation=True, max_length=self.max_target_length)

        return {
            'input_ids_list': [enc['input_ids'] for enc in input_encodings],
            'attention_mask_list': [enc['attention_mask'] for enc in input_encodings],
            'labels': target_encoding['input_ids']
        }
    
    
    
    def collate_fn_fid(batch, tokenizer, max_docs_per_item=10, max_input_length=512, max_target_length=128):
        
        actual_max_docs_in_batch = max(len(item['input_ids_list']) for item in batch) if batch else 0
        max_docs_this_batch = min(actual_max_docs_in_batch, max_docs_per_item)

        max_len_input = max_input_length
        max_len_target = max_target_length

        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        pad_token_id = tokenizer.pad_token_id
        label_pad_token_id = -100

        for item in batch:
            item_input_ids = []
            item_attention_masks = []

            # Process up to max_docs_this_batch, handling items with fewer docs
            num_docs_to_process = min(len(item['input_ids_list']), max_docs_this_batch)

            # Pad up to max_docs_per_item for consistent tensor shapes across batches
            for i in range(max_docs_per_item):
                if i < num_docs_to_process:
                    input_ids = item['input_ids_list'][i][:max_len_input]
                    attention_mask = item['attention_mask_list'][i][:max_len_input]

                    padding_length = max_len_input - len(input_ids)
                    input_ids = input_ids + ([pad_token_id] * padding_length)
                    attention_mask = attention_mask + ([0] * padding_length)
                else:
                    # Pad with empty docs if item has < max_docs_per_item
                    input_ids = [pad_token_id] * max_len_input
                    attention_mask = [0] * max_len_input

                item_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                item_attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))

            if len(item_input_ids) != max_docs_per_item:
                print(f"Warning: Mismatch in expected docs {max_docs_per_item} vs actual {len(item_input_ids)}")

            all_input_ids.append(torch.stack(item_input_ids))
            all_attention_masks.append(torch.stack(item_attention_masks))

            labels = item['labels'][:max_len_target]
            label_padding_length = max_len_target - len(labels)
            padded_labels = labels + ([label_pad_token_id] * label_padding_length)
            all_labels.append(torch.tensor(padded_labels, dtype=torch.long))

        if not all_input_ids:
            return {
                'input_ids': torch.empty(0, max_docs_per_item, max_len_input, dtype=torch.long),
                'attention_mask': torch.empty(0, max_docs_per_item, max_len_input, dtype=torch.long),
                'labels': torch.empty(0, max_len_target, dtype=torch.long)
            }

        batch_input_ids = torch.stack(all_input_ids)
        batch_attention_masks = torch.stack(all_attention_masks)
        batch_labels = torch.stack(all_labels)

        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_masks,
            'labels': batch_labels
        }


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Augment datasets with retrieved documents")
        parser.add_argument(
            "--datasets",
            type=str,
            nargs="+",
            required=True,
            help="List of paths to the datasets to process (separated by space)."
        )
        parser.add_argument(
            "--method",
            type=str,
            required=True,
            default="BM25",
            help="Retrieval method"
        )
        args = parser.parse_args()
        augmented_dataset(args)
        print("Process completed.")
    except Exception as e:
        print(f"Error in creating the augmented dataset: {e}")

    
    