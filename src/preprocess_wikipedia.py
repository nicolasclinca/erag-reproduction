"""
preprocess_wikipedia.py
Downloads and preprocesses the KILT knowledge source (Wikipedia dump).
Splits each article into passages of max 100 words and saves in JSONL format:
{"id": "wikipedia_id_passage_num", "contents": "title [SEP] passage_text"}

Output: ../data/collection/wikipedia_passages.jsonl

CLI usage:

Full download (~5.9M articles → ~108M passages)
python preprocess_wikipedia.py --collection ../data/collection/wikipedia_passages.jsonl

Limit to N records (for test/debug)
python preprocess_wikipedia.py --max_record 10000 --collection ../data/collection/wiki_small.jsonl

With throttling and custom buffer
python preprocess_wikipedia.py --buffer_size 1000000 --throttle \
    --collection ../data/collection/wikipedia_passages.jsonl
"""

import json
import argparse
import time
import requests


def split_into_passages(text, max_words=100):
    """
    Splits text into passages of up to max_words words (without overlap).
    :param text: the entire document
    :param max_words: maximum number of words for each passage
    :return: list of passages
    """
    words = text.split()
    passages = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        passages.append(chunk)
    return passages


def process_kilt_page(page_json, max_words=100):
    """
    Given a KILT knowledge source page (record),
    segments the "text" field (list of paragraphs) into passages of max_words.
    Returns a list of documents combining the title and passage.
    """
    title = page_json.get("wikipedia_title", "")
    paragraphs = page_json.get("text", [])
    id = page_json.get("wikipedia_id", "")
    docs = []
    passage_counter = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        passages = split_into_passages(para, max_words=max_words)
        for p in passages:
            doc = {"id": str(id) + "_" + str(passage_counter), "contents": title + " [SEP] " + p}
            docs.append(doc)
            passage_counter += 1
    return docs


def process_source_request(args):
    """
    Process an entire dataset from an Internet source
    :param args: argparse arguments containing url, output path (collection), buffer_size, max_record,
        and throttle
    :return: number of processed records
    """
    
    processed = 0
    url = args.url
    out_path = args.collection
    max_records = args.max_record
    buffer_size = args.buffer_size
    throttle = args.throttle
    
    with (requests.get(url, stream=True, timeout=10) as in_file,
            open(out_path, 'w', encoding='utf-8') as out_file):
        in_file.raise_for_status()

        buffer = []
        for i, line in enumerate(in_file.iter_lines(decode_unicode=True)):
            if i >= max_records != 0:
                print("Limit reached")
                break
            try:
                page = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding record {i}: {e}")
                continue  

            docs = process_kilt_page(page, max_words=100)
            buffer.extend(docs)

            if len(buffer) >= buffer_size:
                for doc in buffer:
                    json.dump(doc, out_file)
                    out_file.write("\n")
                processed += len(buffer)
                buffer = []
                print(f"\n+++ Buffer emptied: {processed} record processed +++\n")
                
                if throttle:
                    time.sleep(0.5)

        for doc in buffer:
            json.dump(doc, out_file)
            out_file.write("\n")
        processed += len(buffer)

    if max_records == 0:
        print(f"Preprocessing done: processed {processed} records. Output saved in {out_path}")
        return "all"
    else:
        print("Preprocessing done: " + str(max_records) + " records processed")
        return str(max_records)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Preprocess Wikipedia Dump",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--url", type=str, default="http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json",
                        help="KILT knowledge source URL")
    parser.add_argument("--collection", type=str, default="../data/collection/wikipedia_passages.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--max_record", type=int, default=0,
                        help="Number of record taken from the Wikipedia Dump (for smaller wikipedia dump). Default is 0 (no limit).")
    parser.add_argument("--buffer_size", type=int, default=500000,
                        help="Size of the buffer before writing to disk. Default is 500000.")
    parser.add_argument("--throttle", action="store_true",
                        help="Enable throttling (sleep 0.5s after each buffer flush).")
    args = parser.parse_args()
    process_source_request(args)