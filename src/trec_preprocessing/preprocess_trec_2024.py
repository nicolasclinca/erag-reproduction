import os
import json
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

# Percorso del file Qrels di input (UMBRELA con 52k docs)
INPUT_QRELS_FILE = "qrels.rag24.test-umbrela-all.txt" 

# Percorso del file di output
OUTPUT_FILE = "doc_id_text_mapping.jsonl"

# Nome dell'indice Pyserini (MS MARCO V2.1 Segmented)
INDEX_NAME = 'msmarco-v2.1-doc-segmented'

# 1. CARICAMENTO ID UNICI
def load_unique_docids(filename):
    print(f"Lettura ID da: {filename}...")
    unique_ids = set()
    
    if not os.path.exists(filename):
        print(f"Errore: File {filename} non trovato.")
        exit(1)

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                doc_id = parts[2]
                if "msmarco" in doc_id:
                    unique_ids.add(doc_id)
    
    return list(unique_ids)

# 2. RECUPERO TESTO
def fetch_texts(doc_ids, output_path):
    print(f"Inizializzazione Pyserini con indice: {INDEX_NAME}")
    print("   (Se è la prima volta, scaricherà ~55GB di indice...)")
    
    try:
        searcher = LuceneSearcher.from_prebuilt_index(INDEX_NAME)
    except Exception as e:
        print(f"\n ERRORE INIZIALIZZAZIONE SEARCHER: {e}")
        return

    print(f"Inizio recupero testo per {len(doc_ids)} documenti unici...")
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        success_count = 0
        missing_count = 0
        
        for doc_id in tqdm(doc_ids, desc="Fetching"):
            try:
                # Pyserini fetch
                doc = searcher.doc(doc_id)
                
                if doc:
                    # Estrazione contenuto
                    content = doc.contents()
                    
                    # Creazione oggetto JSON
                    entry = {
                        "docid": doc_id,
                        "text": content
                    }
                    f_out.write(json.dumps(entry) + '\n')
                    success_count += 1
                else:
                    missing_count += 1
                    print(f"Warning: {doc_id} not found in index.")
                    
            except Exception as e:
                print(f"Errore su ID {doc_id}: {e}")
                missing_count += 1

    print("\n" + "="*50)
    print("PROCESSO COMPLETATO")
    print("="*50)
    print(f"Documenti salvati: {success_count}")
    print(f"Documenti non trovati: {missing_count}")
    print(f"File di output: {output_path}")

# MAIN
if __name__ == "__main__":
    # 1. Carica gli ID
    doc_ids = load_unique_docids(INPUT_QRELS_FILE)
    print(f"Trovati {len(doc_ids)} ID univoci nel file Qrels.")
    
    # 2. Fetch
    if len(doc_ids) > 0:
        fetch_texts(doc_ids, OUTPUT_FILE)
    else:
        print("Nessun ID trovato. Controlla il formato del file di input.")