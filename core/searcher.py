#!/usr/bin/env python3
"""
Buscador Semántico Central para RAG.
Calcula el embedding de la consulta con Gemini y busca por similitud coseno
sobre cualquier vector_index.db en milisegundos.
"""

import os
import sys
import json
import sqlite3
import argparse
import urllib.request
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_api_key  # noqa: E402

MODEL = "models/gemini-embedding-001"
DIM = 768

def embed_query(query: str, api_key: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:embedContent?key={api_key}"
    data = json.dumps({
        "model": MODEL,
        "content": {"parts": [{"text": query}]},
        "outputDimensionality": DIM
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res["embedding"]["values"]

def search_vector_db(query: str, db_path: Path, top_k: int = 4):
    if not db_path.exists():
        raise FileNotFoundError(f"No existe la base de vectores: {db_path}")

    api_key = get_api_key(hint=db_path)
    q_vec = np.array(embed_query(query, api_key), dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, title, url, text, embedding FROM chunks")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return []

    chunk_ids = [r[0] for r in rows]
    titles = [r[1] for r in rows]
    urls = [r[2] for r in rows]
    texts = [r[3] for r in rows]

    embs = np.array([np.frombuffer(r[4], dtype=np.float32) for r in rows])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_norm = embs / norms

    scores = np.dot(embs_norm, q_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "chunk_id": chunk_ids[idx],
            "title": titles[idx],
            "url": urls[idx],
            "score": float(scores[idx]),
            "text": texts[idx]
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Buscador semántico central para RAG.")
    parser.add_argument("--db", "-d", type=str, required=True, help="Ruta de vector_index.db a consultar.")
    parser.add_argument("query", type=str, help="Pregunta o concepto a buscar.")
    parser.add_argument("--top", "-k", type=int, default=4, help="Número de resultados a devolver.")
    parser.add_argument("--json", action="store_true", help="Salida en JSON crudo.")
    args = parser.parse_args()

    results = search_vector_db(args.query, Path(args.db), top_k=args.top)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    print(f"\n🔎 Resultados para: \"{args.query}\" (Top {len(results)})\n" + "="*60)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['title']} (Similitud: {r['score']:.4f})")
        print(f"    URL: {r['url']} | ID: {r['chunk_id']}")
        print(f"    Texto:\n    {r['text']}\n")

if __name__ == "__main__":
    main()
