#!/bin/bash


NUM_SHARDS=120
CORPUS="data/collection"
ENCODER="BAAI/bge-base-en-v1.5"
# facebook/dpr-ctx_encoder-multiset-base
# castorini/tct_colbert-v2-hnp-msmarco
OUTPUT_DIR="indexes/wiki-bge-118m"

export JAVA_TOOL_OPTIONS="-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"
export OMP_NUM_THREADS=8

# rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

echo "AVVIO INDICIZZAZIONE CON $ENCODER"
echo "Totale parti: $NUM_SHARDS"

for ((i=0; i<NUM_SHARDS; i++))
do
    echo "=================================================="
    echo "ELABORAZIONE PARTE $i di $NUM_SHARDS"
    echo "=================================================="

    python -m pyserini.encode \
      input   --corpus $CORPUS \
              --fields text \
              --shard-id $i \
              --shard-num $NUM_SHARDS \
      output  --embeddings temp_shard_$i \
      encoder --encoder $ENCODER \
              --fields text \
              --batch 128 \
              --fp16 || { echo "ERRORE CRITICO NELLA PARTE $i"; exit 1; }

    python -m pyserini.index.faiss \
      --input temp_shard_$i \
      --output $OUTPUT_DIR/part_$i \
      --pq \
      --pq-m 64 \
      --pq-nbits 8 || { echo "ERRORE INDEXING PARTE $i"; exit 1; }

    echo "Pulizia file temporanei parte $i..."
    rm -rf temp_shard_$i

    echo "Parte $i completata. Spazio libero rimasto:"
    df -h . | tail -1 | awk '{print $4}'
done

echo "TUTTO FINITO CON SUCCESSO!"
