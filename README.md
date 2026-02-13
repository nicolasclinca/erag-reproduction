# «Quis Aestimabit Ipsas Aestimationes?»: A Reproducibility and Benchmarking Study for RAG Evaluation Methods in Information Retrieval Systems

This repository contains the code for the SIGIR ’26 paper titled **«Quis Aestimabit Ipsas Aestimationes?»: A Reproducibility and Benchmarking Study for RAG Evaluation Methods in Information Retrieval Systems**.[LINK PAPER]


Retrieval-Augmented Generation (RAG) systems combine a **retriever**, which selects documents from an external knowledge source (e.g., Wikipedia), and a **generator** (LLM), which produces an answer conditioned on the query and the retrieved documents. Accurately evaluating RAG systems, and in particular **isolating the retriever’s contribution**, is challenging.

Among recent proposals, **eRAG** (Salemi & Zamani, SIGIR ’24) assigns a score to each retrieved document by running the generator **document-by-document**: the generator answers using the query and a single retrieved document, the output is scored with the downstream metric against the ground truth, and these per-document scores are then aggregated using standard IR-style metrics. While the original study reports strong correlation with end-to-end RAG performance and efficiency advantages, incorporating LLMs into evaluation raises concerns regarding **reproducibility, replicability, and generalizability**.

This codebase provides (i) a complete pipeline to reproduce the core eRAG experimental setting, and (ii) a benchmarking extension to analyze eRAG behavior across additional retrievers and (iii) to assess its ability to rank retrievers according to end-to-end RAG performance, as discussed in the paper. [LINK PAPER]

---

## Repository structure

- `environment.yml`: recommended Conda environment (project root)
- `indexing.sh`: helper script for sharded dense indexing (project root)
- `src/`: all Python scripts (currently all files live in this single folder)

---

## Environment setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate <ENV_NAME>
```

Notes:
- Some components require Java (PySerini/Lucene).
- GPU is strongly recommended for FiD inference and for some indexing configurations.

---

## Data prerequisites

### KILT datasets (queries + gold answers)
Download the KILT task datasets from the official repository:
- https://github.com/facebookresearch/KILT/tree/main

This code expects standard KILT `.jsonl` files containing (at least) `id`, `input`, and `output`.

### Knowledge source (Wikipedia via KILT)
The passage collection is built from the KILT knowledge source using `src/preprocess_wikipedia.py`.

---

## What the pipeline produces

For each dataset (e.g., `nq`) you will typically obtain:

- A passage collection: `data/collection/wikipedia_passages.jsonl`
- One or more retrieval run files (CSV): `input_runs/<dataset>/<run>.csv`
- A merged file of all evaluated pairs: `input_runs/<dataset>/<dataset>_all_runs.csv`
- Model-based eRAG qrels: `input_runs/<dataset>/<dataset>_all_qrels.csv`
- Assessed run files: `*_assessed_run.csv` (run + relevance)
- Retrieval metric outputs: `*_retrieval_metrics.csv`, `*_mean_metrics.csv`
- Downstream (end-to-end) outputs: `*_downstream.csv`, `*_mean_downstream.csv`
- Correlation outputs: `*_correlations.csv`

---

## Reproducing the experiments (recommended order)

The workflow has two phases: **Preparation** (mostly one-time) and **Evaluation** (repeat per dataset / retriever).

### Phase A — Preparation

#### A1) Build the passage collection (Wikipedia)
```bash
python src/preprocess_wikipedia.py \
  --collection data/collection/wikipedia_passages.jsonl
```

#### A2) Build retrieval indexes
You can build one or more indexes depending on the retrievers you want to evaluate.

**BM25 (PySerini Lucene)**
```bash
python -m pyserini.index.lucene \
  -collection JsonCollection \
  -input data/collection \
  -index indexes/bm25_index \
  -generator DefaultLuceneDocumentGenerator \
  -threads 8 \
  -storePositions -storeDocvectors -storeRaw
```

**Contriever (single FAISS OPQ+IVF-PQ index)**
```bash
python src/build_contriever_indexes.py \
  --collection data/collection/wikipedia_passages.jsonl \
  --faiss_index_dir indexes/contriever_index \
  --tune \
  --resume
```

**Dense sharded FAISS (DPR/BGE/TCT, PySerini-style)**
Use the provided script as a reference:
```bash
bash indexing.sh
```

For dense-sharded retrieval, you also need a Lucene **docstore** (docid -> raw -> contents):
```bash
python -m pyserini.index.lucene \
  -collection JsonCollection \
  -input data/collection \
  -index indexes/wiki_docstore_lucene \
  -generator DefaultLuceneDocumentGenerator \
  -threads 8 \
  -storeRaw
```

#### A3) Download KILT datasets
Place the `.jsonl` datasets under `data/` (or any path; scripts take `--datasets` explicitly).

#### A4) Build augmented datasets (for FiD training)
Build a retrieved-context dataset for training FiD-T5:
```bash
python src/dataset_builder.py \
  --datasets data/nq-train-kilt.jsonl \
  --method bm25 \
  --bm25_index_dir indexes/bm25_index \
  --k 50
```

#### A5) Train (or load) FiD-T5
Train:
```bash
python src/fid_t5.py train \
  --augmented_datasets data/nq-train-kilt-augmented-bm25.json \
  --model_dir models/fid_t5 \
  --model_name t5-small \
  --num_epochs 10 \
  --per_device_batch_size 1 \
  --effective_batch_size 64 \
  --amp \
  --save_every_epoch
```

You can resume with `--resume_from <.../training_state.pt>`.

---

### Phase B — Evaluation (per dataset)

Assume you create a per-dataset folder:
```bash
mkdir -p input_runs/nq
```

#### B1) Build retrieval run files (one per retriever)
Runs should be in **CSV** format for the full evaluation pipeline.

**BM25**
```bash
python src/build_retrieval_run.py \
  --datasets data/nq-dev-kilt.jsonl \
  --method bm25 \
  --bm25_index_dir indexes/bm25_index \
  --k 50 \
  --format csv \
  --output input_runs/nq/nq_bm25.csv \
  --overwrite
```

**Contriever**
```bash
python src/build_retrieval_run.py \
  --datasets data/nq-dev-kilt.jsonl \
  --method contriever \
  --faiss_index indexes/contriever_index/ivfpq_opq_contriever.faiss \
  --collection data/collection/wikipedia_passages.jsonl \
  --nprobe 64 \
  --k 50 \
  --format csv \
  --output input_runs/nq/nq_contriever.csv \
  --overwrite
```

**Dense sharded (bge/dpr/tct)**
```bash
python src/build_retrieval_run.py \
  --datasets data/nq-dev-kilt.jsonl \
  --method bge \
  --dense_index_root_dir indexes/wiki-bge-118m \
  --docstore_index_dir indexes/wiki_docstore_lucene \
  --k 50 \
  --dense_threads 8 \
  --dense_encode_batch_size 32 \
  --format csv \
  --output input_runs/nq/nq_bge.csv \
  --overwrite
```

Run schema:
- `query_id, doc_id, score, run_id`

#### B2) Merge all runs (build the union of evaluated pairs)
```bash
python src/merge_all_runs.py \
  --input_folder input_runs/nq \
  --overwrite
```

Output:
- `input_runs/nq/nq_all_runs.csv` with columns `query_id, doc_id` (deduplicated).

#### B3) Build eRAG qrels for all (query_id, doc_id) pairs
This runs FiD document-by-document and stores model-based relevance:

```bash
python src/build_all_erag_qrels.py \
  --all_runs input_runs/nq/nq_all_runs.csv \
  --datasets data/nq-dev-kilt.jsonl \
  --collection data/collection/wikipedia_passages.jsonl \
  --model_dir models/fid_t5 \
  --metric em \
  --overwrite
```

Output:
- `input_runs/nq/nq_all_qrels.csv` with columns `query_id, doc_id, relevance` (only `relevance > 0` rows are stored).

The script is incremental: it reuses caches from an existing `*_all_qrels.csv` and any `*_assessed_run.csv` already present in the same folder.

#### B4) Build assessed runs (attach relevance to each run)
```bash
python src/build_all_assessed_runs.py \
  --all_qrels input_runs/nq/nq_all_qrels.csv \
  --input_folder input_runs/nq \
  --overwrite
```

Output per run:
- `*_assessed_run.csv` with columns `query_id, doc_id, score, run_id, relevance`

#### B5) Evaluate retrieval metrics (IR metrics over assessed runs)
Binary relevance (typical with EM/accuracy-based relevance):
```bash
python src/evaluate_all_retrieval_metrics.py \
  --input_folder input_runs/nq \
  --k_values 10 30 50 \
  --overwrite
```

If you use continuous relevance (e.g., `--metric f1` when building qrels), enable:
```bash
python src/evaluate_all_retrieval_metrics.py \
  --input_folder input_runs/nq \
  --k_values 10 30 50 \
  --continuous_relevance \
  --overwrite
```

#### B6) Evaluate end-to-end downstream QA (FiD with top-k contexts)
```bash
python src/evaluate_all_downstream.py \
  --input_folder input_runs/nq \
  --datasets data/nq-dev-kilt.jsonl \
  --collection data/collection/wikipedia_passages.jsonl \
  --model_dir models/fid_t5 \
  --metric em \
  --k_values 10 30 50 \
  --overwrite
```

#### B7) Evaluate correlations (retrieval metrics vs downstream)
```bash
python src/evaluate_correlations.py \
  --input_folder input_runs/nq \
  --overwrite
```

---

## Notes on retrievers implemented

- **BM25** (`src/bm25_retriever.py`): uses a pre-built Lucene index via PySerini.
- **Contriever** (`src/contriever_retriever.py` + `src/build_contriever_indexes.py`): FAISS OPQ+IVF-PQ; FAISS ids map to JSONL line indices and are resolved to the original `id`.
- **Dense sharded (DPR/BGE/TCT)** (`src/dense_sharded_retriever.py`): searches all shards, merges per-query top-k, resolves `rid -> docid` via per-shard `docid` files (LRU cached), then resolves `docid -> contents` via a Lucene docstore.

---

## Citation

If you use this repository, please cite:
- our SIGIR’26 paper [LINK PAPER]
- eRAG (Salemi & Zamani, SIGIR ’24) [LINK PAPER]
