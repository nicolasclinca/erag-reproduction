# «Quis Aestimabit Ipsas Aestimationes?»: A Reproducibility and Benchmarking Study for RAG Evaluation Methods in Information Retrieval Systems

This repository contains the code for the SIGIR ’26 paper titled **«Quis Aestimabit Ipsas Aestimationes?»: A Reproducibility and Benchmarking Study for RAG Evaluation Methods in Information Retrieval Systems**.[LINK PAPER]


Retrieval-Augmented Generation (RAG) systems combine a **retriever**, which selects documents from an external knowledge source (e.g., Wikipedia), and a **generator** (LLM), which produces an answer conditioned on the query and the retrieved documents. Accurately evaluating RAG systems, and in particular **isolating the retriever’s contribution**, is challenging.

Among recent proposals, **eRAG** (Salemi & Zamani, SIGIR ’24) assigns a score to each retrieved document by running the generator **document-by-document**: the generator answers using the query and a single retrieved document, the output is scored with the downstream metric against the ground truth, and these per-document scores are then aggregated using standard IR-style metrics. While the original study reports strong correlation with end-to-end RAG performance and efficiency advantages, incorporating LLMs into evaluation raises concerns regarding **reproducibility, replicability, and generalizability**.

This codebase provides (i) a complete pipeline to reproduce the core eRAG experimental setting, and (ii) a benchmarking extension to analyze eRAG behavior across additional retrievers and (iii) to assess its ability to rank retrievers according to end-to-end RAG performance, as discussed in the paper. [LINK PAPER]

## What this repository provides

- Wikipedia (knowledge source) preprocessing into a **passage collection**.
- Indexing and retrieval (sparse or dense, depending on the configuration).
- Generation and evaluation:
  - **eRAG**: per-document scoring followed by aggregation into IR-style metrics.
  - **End-to-end RAG**: generation with top-*k* documents and downstream scoring.
- Utilities for analysis and comparison (e.g., correlation between aggregated eRAG scores and end-to-end performance; system-level analysis).

## Repository layout (indicative)

- `environment.yml`: recommended Conda environment
- `src/`: main code (preprocess, indexing, retrieval, eRAG, end-to-end, evaluation)
- `data/`: datasets and collection
- `indexing.sh`: indexing helper 
- `logs/`: scoring and evaluation outputs

## Documentation

### Environment (recommended)

```bash
conda env create -f environment.yml
conda activate <ENV_NAME>
```
