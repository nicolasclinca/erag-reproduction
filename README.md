# eRAG_v1.0

## Links
KILT Benchmark: https://github.com/facebookresearch/KILT/tree/main <br>
eRAG: https://github.com/alirezasalemi7/eRAG/tree/main <br>


## 1. Clone the repository
```bash
git clone https://github.com/nicolasclinca/eRAG.git 
```

## 2. Create these directories
Enter the eRAG project directory(cd eRAG) and create these directory. The data directory must contain all the datasets
```bash
mkdir -p data/collection, indexes/bm25_index, indexes/faiss_index, models, logs
```

## Download the NQ dataset (e.g., the dev file)
```bash 
wget -O data/nq-dev-kilt.jsonl http://dl.fbaipublicfiles.com/KILT/nq-dev-kilt.jsonl
wget -O data/nq-train-kilt.jsonl http://dl.fbaipublicfiles.com/KILT/nq-train-kilt.jsonl
wget -O data/nq-test_without_answers-kilt.jsonl http://dl.fbaipublicfiles.com/KILT/nq-test_without_answers-kilt.jsonl
```

## Download the Wikipedia Dump
```bash
wget -O data/wikipedia_dump.jsonl http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
```

## 3. Create a conda environment
```bash
conda env create -f environment.yml
```
Activate the environment
```bash
conda activate nome_ambiente
```

## 4. How to execute the code
1. Execute "preprocess_wikipedia.py" file
```bash
python preprocess_wikipedia.py
```
2. For Pyserini BM25 index run this code:
```bash
python -m pyserini.index.lucene -collection JsonCollection -input ../data/collection -index ../indexes/bm25_index -generator DefaultLuceneDocumentGenerator -threads 8 -storePositions -storeDocvectors -storeRaw
```
Execute "build_indexes.py" file. It contains the code for Contriever (MEGLIO NON AVVIARLO, CONTRIEVER E' DA RIVEDERE) <br>
3. Execute "retrieval_models.py" file <br>
3.1 (Optional) Execute "gemini.py" file, it needs your gemini api key <br>
4. Execute "data_loader.py" file <br>
5. Execute "train.py" file. Use the flag --h for help (change parameters) <br>
```bash
python train.py --h
```
6. Execute "evaluation.py" file. Use the flag --h for help (change parameters) <br> 
