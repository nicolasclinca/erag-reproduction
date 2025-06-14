import json
import torch
from torch.utils.data import Dataset
from retrieval_models import retrieve_documents
import argparse


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


def retrieval_results(filename):
    expected_outputs = load_all_nq_expected_outputs(filename)

    queries = list(expected_outputs.keys())
    print(f"Loaded {len(queries)} queries")
    
    # For each query, retireval_results[query] = [doc1, doc2, ..., doc50]
    retrieve_res = {query: retrieve_documents(query, method='BM25', k=50) for query in queries}
    return expected_outputs, retrieve_res


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
    # Augment the NQ dataset with retrieved documents
    expected_outputs_train = {}
    retrieve_res_train = {}
    filename_train = args.filename_train
    expected_outputs_train, retrieve_res_train = retrieval_results(filename=filename_train)
    augmented_dataset_train = augment_with_retrieved_documents(expected_outputs_train, retrieve_res_train)
    
    expected_outputs_val= {}
    retrieve_res_val = {}
    filename_val = args.filename_val
    expected_outputs_val, retrieve_res_val = retrieval_results(filename=filename_val)
    augmented_dataset_val = augment_with_retrieved_documents(expected_outputs_val, retrieve_res_val)

    # Save the augmented dataset to a new file for training
    with open('../data/augmented_train.json', 'w') as f:
        json.dump(augmented_dataset_train, f, indent=4)
    print(f"Augmented train dataset saved with {len(augmented_dataset_train)} samples.")
    
    with open('../data/augmented_dev.json', 'w') as f:
        json.dump(augmented_dataset_val, f, indent=4)
    print(f"Augmented validation dataset saved with {len(augmented_dataset_val)} samples.")


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
     

if __name__=="__main__":
    try:
        parser = argparse.ArgumentParser(description="Data Loading")
        parser.add_argument("--filename_train", type=str, required=True, default="../data/nq-train-kilt.jsonl",
                        help="Train file name")
        parser.add_argument("--filename_val", type=str, required=True, default="../data/nq-dev-kilt.jsonl",
                        help="Validation file name")
        args = parser.parse_args()
        augmented_dataset(args)  
        print("Dataset augmented and saved.")
    except:
        "Errore nella creazione del dataset aumentato."
    
    