"""
fid_t5.py
Training e inferenza di un modello T5 con architettura Fusion-in-Decoder (FiD).
FiD: encoder processa ogni documento separatamente, poi concatena le rappresentazioni
prima del decoder per generare la risposta finale.

Uso CLI:

Training
python fid_t5.py train --augmented_datasets ../data/train_augmented.json \
    --model_dir ./models/fid_t5 \
    --model_name t5-small \
    --num_epochs 10 \
    --per_device_batch_size 1 \
    --effective_batch_size 64

Inferenza
python fid_t5.py generate --model_dir ./models/fid_t5 \
    --input_json queries_docs.json \
    --output_json predictions.json \
    --num_beams 4
"""

import os
import json
import math
import time
import random
import argparse
import gc
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from transformers import (T5Tokenizer, T5ForConditionalGeneration, get_constant_schedule_with_warmup)
from transformers.modeling_outputs import BaseModelOutput


# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Dataset e Collate FiD
# =========================
class QA_Dataset_FiD(Dataset):
    """
    Dataset per augmented_data nel formato:
      {
        "query": str,
        "gold_answer": str,            # può mancare in inferenza
        "retrieved_docs": List[str]
      }
    """
    def __init__(
        self,
        augmented_data: List[Dict[str, Any]],
        require_answer: bool = True,
        strip_empty_docs: bool = True
    ):
        self.samples = []
        dropped = 0

        for ex in augmented_data:
            query = ex.get("query", None)
            docs = ex.get("retrieved_docs", None)
            answer = ex.get("gold_answer", None)

            # Normalizza docs
            if isinstance(docs, str):
                docs = [docs]
            if strip_empty_docs and isinstance(docs, list):
                docs = [str(d) for d in docs if isinstance(d, str) and d.strip() != ""]

            if not isinstance(query, str) or not isinstance(docs, list) or len(docs) == 0:
                dropped += 1
                continue

            if require_answer and not (isinstance(answer, str) and answer.strip() != ""):
                dropped += 1
                continue

            self.samples.append({
                "query": query,
                "docs": docs,
                "answer": (answer.strip() if isinstance(answer, str) else None)
            })

        if dropped > 0:
            print(f"[QA_Dataset_FiD] Scartati {dropped} esempi malformati o senza risposta.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Restituiamo stringhe; la tokenizzazione avviene nel collate
        return self.samples[idx]

    @staticmethod
    def collate_fn_fid(
        batch: List[Dict[str, Any]],
        tokenizer: T5Tokenizer,
        max_docs_per_item: int = 50,
        max_input_len: int = 256,
        max_target_len: int = 64
    ) -> Dict[str, torch.Tensor]:
        """
        Restituisce tensori FiD con shape:
          - input_ids: (B, max_docs_per_item, max_input_len)
          - attention_mask: (B, max_docs_per_item, max_input_len)
          - labels: (B, max_target_len) con pad -> -100
        """
        if not batch:
            return {
                "input_ids": torch.empty(0, max_docs_per_item, max_input_len, dtype=torch.long),
                "attention_mask": torch.empty(0, max_docs_per_item, max_input_len, dtype=torch.long),
                "labels": torch.empty(0, max_target_len, dtype=torch.long),
            }

        pad_id = tokenizer.pad_token_id
        label_pad_id = -100

        actual_max_docs_in_batch = max(len(ex["docs"]) for ex in batch)
        max_docs_this_batch = min(actual_max_docs_in_batch, max_docs_per_item)

        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for ex in batch:
            query = ex["query"]
            docs = ex["docs"]
            answer = ex.get("answer", None)

            num_docs_to_process = min(len(docs), max_docs_this_batch)

            if num_docs_to_process > 0:
                texts = [f"question: {query} context: {docs[i]}" for i in range(num_docs_to_process)]
                enc = tokenizer(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=max_input_len,
                    return_tensors="pt",
                )
                in_ids = enc.input_ids            # (n_i, L)
                attn = enc.attention_mask         # (n_i, L)
            else:
                in_ids = torch.empty(0, max_input_len, dtype=torch.long)
                attn = torch.empty(0, max_input_len, dtype=torch.long)

            if num_docs_to_process < max_docs_per_item:
                pad_docs = max_docs_per_item - num_docs_to_process
                pad_ids_block = torch.full((pad_docs, max_input_len), pad_id, dtype=torch.long)
                pad_attn_block = torch.zeros((pad_docs, max_input_len), dtype=torch.long)
                if num_docs_to_process == 0:
                    in_ids = pad_ids_block
                    attn = pad_attn_block
                else:
                    in_ids = torch.cat([in_ids, pad_ids_block], dim=0)
                    attn = torch.cat([attn, pad_attn_block], dim=0)

            all_input_ids.append(in_ids)         # (max_docs_per_item, L)
            all_attention_masks.append(attn)     # (max_docs_per_item, L)

            if isinstance(answer, str) and answer.strip() != "":
                lab = tokenizer(
                    answer,
                    truncation=True,
                    padding="max_length",
                    max_length=max_target_len,
                    return_tensors="pt",
                ).input_ids.squeeze(0)           # (T)
                lab[lab == pad_id] = label_pad_id
            else:
                lab = torch.full((max_target_len,), label_pad_id, dtype=torch.long)

            all_labels.append(lab)

        batch_input_ids = torch.stack(all_input_ids, dim=0)          # (B, max_docs_per_item, L)
        batch_attention_masks = torch.stack(all_attention_masks, 0)  # (B, max_docs_per_item, L)
        batch_labels = torch.stack(all_labels, dim=0)                # (B, T)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_masks,
            "labels": batch_labels,
        }


# =========================
# FiD helpers
# =========================
def fid_encode_concat(
    model: T5ForConditionalGeneration,
    input_ids_batch: torch.Tensor,       # (B, N, L)
    attention_mask_batch: torch.Tensor,  # (B, N, L)
) -> tuple[BaseModelOutput, torch.Tensor]:
    """
    Esegue l'encoder T5 per-doc e concatena lungo la dimensione sequenza (FiD).
    Restituisce:
      - encoder_outputs: BaseModelOutput(last_hidden_state=(B, N*L, d))
      - encoder_attention_mask: (B, N*L)
    """
    bsz, n_docs, seq_len = input_ids_batch.shape

    # (B*N, L)
    input_ids_enc = input_ids_batch.view(bsz * n_docs, seq_len)
    attention_mask_enc = attention_mask_batch.view(bsz * n_docs, seq_len)

    encoder_out = model.encoder(
        input_ids=input_ids_enc,
        attention_mask=attention_mask_enc,
        return_dict=True
    )
    last_hidden = encoder_out.last_hidden_state  # (B*N, L, d)

    # -> (B, N, L, d) -> (B, N*L, d)
    d_model = last_hidden.size(-1)
    last_hidden = last_hidden.view(bsz, n_docs, seq_len, d_model).reshape(bsz, n_docs * seq_len, d_model)

    encoder_outputs_fid = BaseModelOutput(last_hidden_state=last_hidden)
    encoder_attention_mask = attention_mask_enc.view(bsz, n_docs, seq_len).reshape(bsz, n_docs * seq_len)

    return encoder_outputs_fid, encoder_attention_mask


# =========================
# Training
# =========================
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Opzioni memoria/stabilità
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()

    model.to(device)

    # Datasets e Dataloaders
    with open(args.augmented_datasets, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print("Loading dataset...")
    train_ds = QA_Dataset_FiD(train_data, require_answer=True)

    collate = lambda batch: QA_Dataset_FiD.collate_fn_fid(
        batch,
        tokenizer=tokenizer,
        max_docs_per_item=args.max_docs_per_item,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    # Ottimizzatore e scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.gradient_accumulation_steps is None:
        if args.effective_batch_size % args.per_device_batch_size != 0:
            raise ValueError("effective_batch_size deve essere divisibile per per_device_batch_size "
                             "oppure specifica --gradient_accumulation_steps.")
        grad_accum = args.effective_batch_size // args.per_device_batch_size
    else:
        grad_accum = args.gradient_accumulation_steps

    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = updates_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
    )

    print("----- Config -----")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Per-device batch size: {args.per_device_batch_size}")
    print(f"Effective batch size: {args.effective_batch_size} (grad_accum={grad_accum})")
    print(f"Max docs per item: {args.max_docs_per_item}")
    print(f"Max input len: {args.max_input_len} | Max target len: {args.max_target_len}")
    print(f"LR: {args.lr} | Weight decay: {args.weight_decay}")
    print(f"Warmup ratio: {args.warmup_ratio} -> {warmup_steps} steps")
    print(f"Total optimizer steps (approx): {total_steps}")
    print(f"AMP: {args.amp} | Grad checkpointing: {args.grad_checkpointing}")
    print("------------------")

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    global_step = 0
    os.makedirs(args.model_dir, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss_sum, epoch_count = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Epoch {epoch}/{args.num_epochs}")
        for step_idx, batch in pbar:
            input_ids = batch["input_ids"].to(device)       # (B, N, L)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)             # (B, T)

            encoder_outputs, enc_attn_mask = fid_encode_concat(model, input_ids, attention_mask)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and device.type == "cuda")):
                outputs = model(
                    labels=labels,
                    encoder_outputs=encoder_outputs,
                    attention_mask=enc_attn_mask,
                    return_dict=True,
                )
                loss = outputs.loss

            if loss is None or not torch.isfinite(loss):
                continue

            loss_to_backprop = loss / grad_accum
            if args.amp and device.type == "cuda":
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            epoch_loss_sum += loss.item()
            epoch_count += 1

            # optimizer step ogni grad_accum
            if step_idx % grad_accum == 0:
                if args.max_grad_norm is not None:
                    if args.amp and device.type == "cuda":
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if args.amp and device.type == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                pbar.set_postfix(
                    loss=f"{(epoch_loss_sum / max(epoch_count,1)):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )

        # Flush di eventuali gradienti residui se l'epoch non è multiplo di grad_accum
        if (len(train_loader) % grad_accum) != 0:
            if args.max_grad_norm is not None:
                if args.amp and device.type == "cuda":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.amp and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        avg_train_loss = epoch_loss_sum / max(epoch_count, 1)
        print(f"Epoch {epoch} done. Train loss: {avg_train_loss:.4f}")

        # Checkpoint per-epoca (opzionale)
        if args.save_every_epoch:
            save_dir = os.path.join(args.model_dir, f"epoch_{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Saved epoch checkpoint to: {save_dir}")
        elif epoch == args.num_epochs:
            # Salva il modello finale
            model.save_pretrained(args.model_dir)
            tokenizer.save_pretrained(args.model_dir)
            print(f"Saved final model to: {args.model_dir}")

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Training finished.")


# =========================
# Generazione (Inferenza)
# =========================
def t5_fid_generator(
    queries_and_documents: Dict[str, List[str]],
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    device: torch.device,
    max_input_len: int = 256,
    max_output_len: int = 64,
    num_beams: int = 4,
    **generate_kwargs
) -> Dict[str, str]:
    """
    FiD generation su un dizionario {query: [doc1, doc2, ...]}.
    """
    model.eval()
    results = {}

    print(f"Generating answers for {len(queries_and_documents)} queries...")
    t0 = time.time()

    for i, (query, docs) in enumerate(queries_and_documents.items(), start=1):
        if not docs:
            results[query] = "Error: No documents provided."
            continue

        # Tokenizza i documenti della query con pad dinamico
        texts = [f"question: {query} context: {doc}" for doc in docs]
        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_input_len,
            return_tensors="pt",
        )
        input_ids = enc.input_ids.to(device)          # (N, L_i)
        attention_mask = enc.attention_mask.to(device)

        with torch.inference_mode():
            # Encoder per-doc
            enc_out = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            last_hidden = enc_out.last_hidden_state   # (N, L, d)
            N, L, d = last_hidden.shape

            # Concat FiD: (1, N*L, d)
            enc_hidden_concat = last_hidden.reshape(1, N * L, d)
            enc_attn_mask = attention_mask.reshape(1, N * L)

            encoder_outputs_for_generate = BaseModelOutput(last_hidden_state=enc_hidden_concat)

            # Decoding
            gen_ids = model.generate(
                encoder_outputs=encoder_outputs_for_generate,
                attention_mask=enc_attn_mask,
                num_beams=num_beams,
                max_new_tokens=max_output_len,
                early_stopping=True,
                **generate_kwargs
            )

        out_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        results[query] = out_text

        if i % 100 == 0:
            print(f"  Generated {i}/{len(queries_and_documents)}...")

    print(f"Done. Time: {time.time() - t0:.2f}s")
    return results


def load_queries_docs_from_json(path: str) -> Dict[str, List[str]]:
    """
    Carica un JSON di inferenza nel formato:
      {
        "query 1": ["doc1", "doc2", ...],
        "query 2": ["doc1", ...],
        ...
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned = {}
    for q, docs in data.items():
        if isinstance(docs, str):
            cleaned[q] = [docs]
        elif isinstance(docs, list):
            cleaned[q] = [str(d) for d in docs]
        else:
            cleaned[q] = []
    return cleaned


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Unified T5 FiD Training and Inference",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train a T5 FiD model")
    p_train.add_argument("--augmented_datasets", type=str, required=True, help="Path to train JSON")
    p_train.add_argument("--model_dir", type=str, default="./models/fid_t5", help="Output dir")
    p_train.add_argument("--model_name", type=str, default="t5-small", help="HF model name or path")
    p_train.add_argument("--num_epochs", type=int, default=10)
    p_train.add_argument("--per_device_batch_size", type=int, default=1)
    p_train.add_argument("--effective_batch_size", type=int, default=64)
    p_train.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p_train.add_argument("--max_docs_per_item", type=int, default=50)
    p_train.add_argument("--max_input_len", type=int, default=256)
    p_train.add_argument("--max_target_len", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=5e-5)
    p_train.add_argument("--weight_decay", type=float, default=1e-2)
    p_train.add_argument("--warmup_ratio", type=float, default=0.05)
    p_train.add_argument("--max_grad_norm", type=float, default=1.0)
    p_train.add_argument("--num_workers", type=int, default=2)
    p_train.add_argument("--amp", action="store_true", help="Enable mixed precision (fp16)")
    p_train.add_argument("--grad_checkpointing", action="store_true")
    p_train.add_argument("--save_every_epoch", action="store_true")
    p_train.add_argument("--seed", type=int, default=42)

    # Generate
    p_gen = subparsers.add_parser("generate", help="Run inference with a trained T5 FiD model")
    p_gen.add_argument("--model_dir", type=str, required=True, help="Path to saved model (e.g., model_dir/epoch_x)")
    p_gen.add_argument("--input_json", type=str, required=True, help="JSON file {query: [docs,...]}")
    p_gen.add_argument("--output_json", type=str, required=True, help="Where to save {query: answer}")
    p_gen.add_argument("--max_input_len", type=int, default=256)
    p_gen.add_argument("--max_new_tokens", type=int, default=64)
    p_gen.add_argument("--num_beams", type=int, default=4)
    p_gen.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "train":
        train(args)

    elif args.command == "generate":
        set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
        model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)
        model.eval()

        qd = load_queries_docs_from_json(args.input_json)
        results = t5_fid_generator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            queries_and_documents=qd,
            max_input_len=args.max_input_len,
            max_output_len=args.max_new_tokens,
            num_beams=args.num_beams,
        )

        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved predictions to {args.output_json}")


if __name__ == "__main__":
    main()