# eRAG_v1.0

## 1. Create a conda environment
```bash
conda env create -f environment.yml
```

## 2. Create a directory for data if it doesn't exist
The data directory must contain all the datasets
```bash
mkdir -p data
```

## Download the NQ dataset (e.g., the dev file)
```bash wget -O data/nq-dev-kilt.jsonl http://dl.fbaipublicfiles.com/KILT/nq-dev-kilt.jsonl
wget -O data/nq-train-kilt.jsonl http://dl.fbaipublicfiles.com/KILT/nq-train-kilt.jsonl
wget -O data/nq-test_without_answers-kilt.jsonl http://dl.fbaipublicfiles.com/KILT/nq-test_without_answers-kilt.jsonl
```

## Download the Wikipedia Dump
```bash
wget -O data/wikipedia_dump.jsonl http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
```
