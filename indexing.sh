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

echo "START INDEXING WITH $ENCODER"
echo "Total parts: $NUM_SHARDS"

for ((i=0; i<NUM_SHARDS; i++))
do
    echo "=================================================="
    echo "PROCESSING PART $i OF $NUM_SHARDS"
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
              --fp16 || { echo "CRITICAL ERROR IN $i"; exit 1; }

    python -m pyserini.index.faiss \
      --input temp_shard_$i \
      --output $OUTPUT_DIR/part_$i \
      --pq \
      --pq-m 64 \
      --pq-nbits 8 || { echo "INDEXING ERROR IN $i"; exit 1; }

    echo "Cleaning temporary files part $i..."
    rm -rf temp_shard_$i

    echo "Part $i completed. Remaining free space:"
    df -h . | tail -1 | awk '{print $4}'
done

echo "SUCCESSFULLY COMPLETED!"
