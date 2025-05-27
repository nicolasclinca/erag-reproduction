import torch
from tqdm.auto import tqdm
from data_loader import QA_Dataset_FiD, data_loading
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from functools import partial
import math
import gc
import os
import argparse


def train(args):

    train_data = "../data/augmented_nq_train.json"
    test_data = "../data/augmented_nq_dev.json"

    # Initialize the tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Create Datasets
    train_dataset = QA_Dataset_FiD(train_data, tokenizer)
    test_dataset = QA_Dataset_FiD(test_data, tokenizer)
    
    # --- DataLoader Setup ---
    per_device_batch_size = 1
    num_epochs = args.num_epochs
    effective_batch_size = 64 # As per paper
    max_docs_per_item = args.max_docs_per_item

    
    # Calculate gradient accumulation steps
    if effective_batch_size % per_device_batch_size != 0:
        raise ValueError("Effective batch size must be divisible by per-device batch size")
    gradient_accumulation_steps = effective_batch_size // per_device_batch_size

    collate_fn_fid = QA_Dataset_FiD.collate_fn_fid
    train_dataloader = DataLoader(train_dataset, batch_size=per_device_batch_size, shuffle=True,
                                collate_fn=partial(collate_fn_fid, tokenizer=tokenizer, max_docs_per_item=max_docs_per_item))
    test_dataloader = DataLoader(test_dataset, batch_size=per_device_batch_size, shuffle=False,
                                collate_fn=partial(collate_fn_fid, tokenizer=tokenizer, max_docs_per_item=max_docs_per_item))
    # --- End DataLoader Setup ---


    # --- Model, Optimizer, and Device Setup ---
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    initial_lr = 5e-5 # As per paper
    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-2) # As per paper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # --- End Model Setup ---


    # --- Warmup and Training Steps Calculation ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if num_update_steps_per_epoch == 0:
        raise ValueError("Train dataloader is effectively empty with gradient accumulation. Check dataset size and batch sizes.")

    total_training_steps = num_epochs * num_update_steps_per_epoch
    warmup_proportion = 0.05 # 5% as per paper
    num_warmup_steps = int(total_training_steps * warmup_proportion)
    global_optimizer_step = 0
    # --- End Warmup Calculation ---


    print(f"--- Configuration ---")
    print(f"Using device: {device}")
    print(f"Per-device batch size: {per_device_batch_size}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Gradient Accumulation steps: {gradient_accumulation_steps}")
    print(f"Max docs per item: {args.max_docs_per_item}")
    print(f"Num Epochs: {args.num_epochs}")
    print(f"Initial Learning Rate: {initial_lr}")
    print(f"Weight Decay: {optimizer.param_groups[0]['weight_decay']}")
    print(f"Train Dataloader size (batches): {len(train_dataloader)}")
    print(f"Optimizer steps per epoch: {num_update_steps_per_epoch}")
    print(f"Total optimizer steps: {total_training_steps}")
    print(f"Warmup optimizer steps: {num_warmup_steps}")
    print(f"--------------------")

        
    model.zero_grad()

    for epoch in range(num_epochs):
        model.train()
        total_loss_accumulated = 0.0
        processed_batches_count = 0

        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):

            input_ids_batch = batch['input_ids'].to(device)
            attention_mask_batch = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if input_ids_batch.numel() == 0:
                print(f"Warning: Skipping empty batch at step {i}")
                continue

            bsz, n_docs, seq_len = input_ids_batch.shape
            target_seq_len = labels.shape[-1]

            if labels.dim() == 3 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            elif labels.dim() != 2:
                print(f"Unexpected labels shape: {labels.shape} at step {i}. Skipping batch.")
                continue

            # --- FiD Forward Pass ---
            try:
                # 1. Prepare and Run Encoder
                input_ids_enc = input_ids_batch.view(bsz * n_docs, seq_len)
                attention_mask_enc = attention_mask_batch.view(bsz * n_docs, seq_len)

                encoder_outputs = model.encoder(
                    input_ids=input_ids_enc,
                    attention_mask=attention_mask_enc,
                    return_dict=True
                )

                # 2. Prepare Inputs for Decoder
                cross_attention_mask = attention_mask_batch.view(bsz, n_docs * seq_len)

                # 3. Run Decoder
                outputs = model(
                    labels=labels,
                    encoder_outputs=encoder_outputs,
                    attention_mask=cross_attention_mask
                )
                loss = outputs.loss

                if loss is None:
                    print(f"Warning: Loss is None at step {i}. Skipping backward/step.")
                    continue

                # --- Scale Loss for Gradient Accumulation ---
                # Normalize loss to average gradients correctly
                scaled_loss = loss / gradient_accumulation_steps
                total_loss_accumulated += loss.item()
                processed_batches_count += 1

                # --- Backward Pass (Accumulate Gradients) ---
                scaled_loss.backward()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM error during forward/backward at step {i}. Skipping batch.")
                    print(f"  Batch shapes: input_ids {input_ids_batch.shape}, labels {labels.shape}")
                    # Clear cache and try to continue
                    del input_ids_batch, attention_mask_batch, labels, batch
                    if 'encoder_outputs' in locals(): del encoder_outputs
                    if 'outputs' in locals(): del outputs
                    if 'loss' in locals(): del loss
                    if 'scaled_loss' in locals(): del scaled_loss
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Runtime error during forward/backward: {e} at step {i}. Skipping batch.")
                    continue
            except Exception as e:
                print(f"General error during forward/backward: {e} at step {i}. Skipping batch.")
                continue

            # --- Optimizer Step ---
            # Check if we have processed enough batches for one optimizer update
            if (i + 1) % gradient_accumulation_steps == 0:

                # --- Optional: Gradient Clipping (Common practice) ---
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # --- Apply Linear Warmup ---
                if global_optimizer_step < num_warmup_steps:
                    lr_scale = float(global_optimizer_step) / float(max(1, num_warmup_steps))
                else:
                    lr_scale = 1.0

                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_lr * lr_scale
                # --- End Linear Warmup ---

                # --- Perform Optimizer Step ---
                optimizer.step()

                # --- Zero Gradients for the next accumulation cycle ---
                optimizer.zero_grad()

                # --- Increment global OPTIMIZER step counter ---
                global_optimizer_step += 1

                if global_optimizer_step % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    recent_loss = total_loss_accumulated / processed_batches_count if processed_batches_count > 0 else 0.0
                    print(f"Epoch: {epoch}, Opt Step: {global_optimizer_step}/{total_training_steps}, LR: {current_lr:.2e}, Avg Epoch Loss So Far: {recent_loss:.4f}")


        # --- End of Epoch ---
        avg_epoch_loss = total_loss_accumulated / processed_batches_count if processed_batches_count > 0 else 0
        print(f"Epoch {epoch} Finished - Average Loss: {avg_epoch_loss:.4f}")

        # --- Save Model Checkpoint Per Epoch ---
        output_dir = f"../models/finetuned_t5_model_fid"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model checkpoint saved to {output_dir}")

    print("Training finished.")
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train a T5 FiD model")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs.Default is 10")
    parser.add_argument("--max_docs_per_item", type=int, default=50,
                        help="Maximum number of documents to consider for each question. Default is 50")
    args = parser.parse_args()
    train(args)
    