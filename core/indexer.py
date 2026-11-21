#!/usr/bin/env python3
"""
Indexador Vectorial Central para RAG.
Lee cualquier chunks.jsonl estándar y genera/actualiza su vector_index.db con Gemini.
"""

import os
import sys
import json
import time
import sqlite3
import argparse
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_api_key  # noqa: E402

MODEL = "models/gemini-embedding-001"
DIM = 768
DEFAULT_BATCH_SIZE = 45

def init_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        source_id TEXT,
        title TEXT,
        url TEXT,
        chunk_index INTEGER,
        source TEXT,
        text TEXT,
        embedding BLOB
    )
    """)
    conn.commit()
    return conn

def embed_batch(texts, api_key, retries=8):
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:batchEmbedContents?key={api_key}"
    requests_payload = [
        {
            "model": MODEL,
            "content": {"parts": [{"text": t}]},
            "outputDimensionality": DIM
        }
        for t in texts
    ]
    data = json.dumps({"requests": requests_payload}).encode("utf-8")
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                embeddings = [e["values"] for e in res.get("embeddings", [])]
                if len(embeddings) == len(texts):
                    return embeddings
                raise ValueError(f"Esperados {len(texts)} embeddings, recibidos {len(embeddings)}")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            if e.code == 429 and attempt < retries - 1:
                wait_time = 62
                print(f"\n[Aviso] Cuota por minuto de Gemini alcanzada. Esperando {wait_time}s...")
                time.sleep(wait_time)
                continue
            if e.code in (500, 503) and attempt < retries - 1:
                time.sleep((2 ** attempt) + 2)
                continue
            raise RuntimeError(f"Error {e.code} de Gemini API: {err_body}")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
                continue
            raise e

def index_file(chunks_file: Path, db_file: Path, batch_size: int = DEFAULT_BATCH_SIZE):
    if not chunks_file.exists():
        raise FileNotFoundError(f"No existe el archivo de chunks: {chunks_file}")

    api_key = get_api_key(hint=db_file)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(db_file)
    cur = conn.cursor()

    cur.execute("SELECT chunk_id FROM chunks")
    indexed_ids = set(row[0] for row in cur.fetchall())
    print(f"Indexados previamente en {db_file.name}: {len(indexed_ids)}")

    pending = []
    with open(chunks_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["chunk_id"] not in indexed_ids:
                pending.append(item)

    total_pending = len(pending)
    print(f"Pendientes por indexar: {total_pending}")
    if total_pending == 0:
        print("Todo actualizado al 100%.")
        return len(indexed_ids)

    processed = 0
    for i in range(0, total_pending, batch_size):
        batch = pending[i : i + batch_size]
        texts = [b["text"] for b in batch]
        vectors = embed_batch(texts, api_key)

        insert_rows = []
        for item, vec in zip(batch, vectors):
            vec_blob = np.array(vec, dtype=np.float32).tobytes()
            insert_rows.append((
                item["chunk_id"],
                item.get("source_id", ""),
                item.get("title", ""),
                item.get("url", ""),
                item.get("chunk_index", 0),
                item.get("source", "general"),
                item["text"],
                vec_blob
            ))

        cur.executemany("""
        INSERT OR REPLACE INTO chunks 
        (chunk_id, source_id, title, url, chunk_index, source, text, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, insert_rows)
        conn.commit()

        processed += len(batch)
        print(f"Progreso: {processed}/{total_pending} ({processed/total_pending*100:.1f}%)")
        if processed < total_pending:
            time.sleep(30)

    cur.execute("SELECT COUNT(*) FROM chunks")
    total_in_db = cur.fetchone()[0]
    print(f"Listo. Total chunks en {db_file}: {total_in_db}")
    return total_in_db

def main():
    parser = argparse.ArgumentParser(description="Indexador vectorial central para RAG.")
    parser.add_argument("--chunks", "-c", type=str, required=True, help="Ruta de chunks.jsonl a indexar.")
    parser.add_argument("--db", "-d", type=str, required=True, help="Ruta de la base vector_index.db destino.")
    parser.add_argument("--batch", "-b", type=int, default=DEFAULT_BATCH_SIZE, help="Tamaño de lote (defecto: 45).")
    args = parser.parse_args()

    index_file(Path(args.chunks), Path(args.db), batch_size=args.batch)

if __name__ == "__main__":
    main()
