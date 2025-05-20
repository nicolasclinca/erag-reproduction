import ujson as json
import argparse

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
        # Split the paragraph into passages of max_words
        passages = split_into_passages(para, max_words=max_words)
        # Concatenate title and passage for each segment, cioè assegna a quel titolo il passaggio a cui è legato
        for p in passages:
            doc = {"id": str(id) + "_" + str(passage_counter), "contents": title + " [SEP] " + p}
            docs.append(doc)
            passage_counter += 1
    return docs


def process_source(in_path, out_path, args, buffer_size=2_000_000):
    """
    Process an entire dataset from an Internet source
    :param in_path: link to the dataset
    :param out_path: path to the output file
    :param buffer_size: size of the buffer to speed up the procedure (default: 2 Millions)
    :param max_records: limit to the number of records to be processed; if 0 (default value), there is no limit
    :return: number of processed records
    """
    max_records = args.max_record
    with (open(in_path, 'r', encoding='utf-8') as in_file,
          open(out_path, 'w', encoding='utf-8') as out_file):

        buffer = []
        for i, line in enumerate(in_file):
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
                    # Save each document as JSON Lines
                    json.dump(doc, out_file)
                    out_file.write("\n")
                buffer = []

        for doc in buffer:
            json.dump(doc, out_file)
            out_file.write("\n")

        if max_records == 0:
            return "all"
        else:
            return str(max_records)


if __name__=="__main__":
    input_path = '../data/wikipedia_dump.jsonl'
    output_path = '../data/collection/wikipedia_passages.jsonl'
    parser = argparse.ArgumentParser(description="Preprocess Wikipedia Dump")
    parser.add_argument("--max_record", type=int, default=0,
                        help="Number of record taken from thw Wikipedia Dump (for smaller wikipedia dump). Default is 0 (no limit).")
    args = parser.parse_args()
    # print("Processed " + process_source(input_path, output_path, max_records=50) + " records")
    process_source(input_path, output_path, args)