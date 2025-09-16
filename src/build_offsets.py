import os, struct, argparse

"""
Uso CLI:
  python build_offsets.py --collection ../data/collection/wikipedia_passages.jsonl \
                          --out ./index_out_full/collection_offsets.u64.bin
"""

def build_offsets(collection_path: str, offsets_path: str) -> int:
    os.makedirs(os.path.dirname(offsets_path) or ".", exist_ok=True)
    count = 0
    with open(collection_path, "rb") as fin, open(offsets_path, "wb") as fout:
        while True:
            pos = fin.tell()
            line = fin.readline()
            if not line:
                break
            fout.write(struct.pack("<Q", pos))  # uint64 little-endian
            count += 1
    size_mb = os.path.getsize(offsets_path) / (1024**2)
    print(f"Offsets written: {count} lines → {size_mb:.1f} MB at {offsets_path}")
    return count

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build offsets file for JSONL (one uint64 per line).")
    ap.add_argument("--collection", required=True, help="Path to JSONL (id, contents).")
    ap.add_argument("--out", required=True, help="Path to write offsets (e.g., ./index_out/collection_offsets.u64.bin).")
    args = ap.parse_args()
    build_offsets(args.collection, args.out)