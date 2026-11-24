#!/usr/bin/env python3
"""
Indexador Vectorial Central para RAG.

Lee cualquier chunks.jsonl estándar y genera/actualiza su vector_index.db.
Elige el motor de embeddings: Gemini (nube) o BGE-M3 local vía Ollama.

  python indexer.py -c chunks.jsonl -d vector_index.db                 # auto
  python indexer.py -c chunks.jsonl -d vector_index.db --provider ollama
  python indexer.py -c chunks.jsonl -d vector_index.db --rebuild       # rehacer desde cero

El proveedor que se use queda grabado en la tabla ``meta`` del índice. Mezclar
motores en un mismo índice daría búsquedas sin sentido (768 vs 1024 dim), así
que reindexar con otro motor exige ``--rebuild`` o una base nueva.
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedders as emb  # noqa: E402

DEFAULT_BATCH_SIZE = 45
GEMINI_PAUSE = 30  # segundos entre lotes: respeta el rate limit de Gemini


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
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
        """
    )
    emb.ensure_meta_table(conn)
    conn.commit()
    return conn


def embed_batch(texts, api_key=None, retries=8):
    """Compatibilidad: embeddings de Gemini para una lista de textos."""
    del retries
    return emb.GeminiEmbedder(api_key=api_key).embed(texts)


def _select_embedder(db_file: Path, meta: dict, provider: str):
    """Devuelve (embedder, nota) respetando el motor con el que se creó el índice."""
    stored = emb.meta_provider(meta) if meta else None
    wanted = (provider or "auto").strip().lower()

    if stored:
        if wanted not in ("auto", stored):
            raise SystemExit(
                f"⚠️  El índice {db_file.name} se creó con el motor '{stored}'.\n"
                f"    Cambiar a '{wanted}' exige reindexar: añade --rebuild o usa otra base."
            )
        embedder = emb.build(stored)
        ok, detail = embedder.available()
        if not ok:
            raise SystemExit(
                f"❌ El índice {db_file.name} es de '{stored}' y ese motor no está disponible: {detail}\n"
                f"    Reindexa con el otro motor: --rebuild --provider "
                f"{'ollama' if stored == 'gemini' else 'gemini'}"
            )
        return embedder, f"{stored} (fijado por el índice: {detail})"

    embedder, note = emb.resolve(wanted)
    return embedder, note


def index_file(
    chunks_file: Path,
    db_file: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    provider: str = "auto",
    rebuild: bool = False,
) -> int:
    if not chunks_file.exists():
        raise FileNotFoundError(f"No existe el archivo de chunks: {chunks_file}")

    meta = emb.read_index_meta(db_file)
    conn = init_db(db_file)
    cur = conn.cursor()

    if rebuild and meta:
        print(f"♻️  --rebuild: se borran los vectores de {db_file.name} (los datos siguen en {chunks_file.name}).")
        cur.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM meta")
        conn.commit()
        meta = {}

    embedder, note = _select_embedder(db_file, meta, provider)
    print(f"🧠 Motor de embeddings: {embedder.name} · modelo {getattr(embedder, 'model', '?')} · {note}")

    cur.execute("SELECT chunk_id FROM chunks")
    indexed_ids = {row[0] for row in cur.fetchall()}
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
        if not meta and getattr(embedder, "dim", None):
            emb.write_index_meta(conn, embedder, note=note)  # sella índices antiguos sin metadatos
        print("Todo actualizado al 100%.")
        return len(indexed_ids)

    processed = 0
    # En modo auto con un índice nuevo, si Gemini falla antes de escribir nada se cambia a local.
    switchable = provider.strip().lower() == "auto" and not meta and embedder.name == "gemini"
    for i in range(0, total_pending, batch_size):
        batch = pending[i : i + batch_size]
        texts = [b["text"] for b in batch]
        try:
            vectors = embedder.embed(texts)
        except emb.ProviderUnavailable as e:
            if switchable and processed == 0 and not cur.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
                local = emb.build("ollama")
                ok, detail = local.available()
                if ok:
                    print(f"\n↩️  {e}\n   → Cambiando a BGE-M3 local ({detail}).")
                    embedder, note = local, f"respaldo local: {detail}"
                    switchable = False
                    try:
                        vectors = embedder.embed(texts)
                    except emb.EmbeddingError as e2:
                        print(f"\n❌ También falló el motor local: {e2}")
                        conn.close()
                        raise SystemExit(2) from e2
                else:
                    print(f"\n❌ {e}\n   Y no hay respaldo local: {detail}")
                    conn.close()
                    raise SystemExit(2) from e
            else:
                print(f"\n❌ {e}")
                print(f"   Progreso guardado: {processed}/{total_pending} chunks (el índice es reanudable).")
                other = "ollama" if embedder.name == "gemini" else "gemini"
                fallback = emb.fallback_db_path(db_file)
                print(f"   Para terminar con el otro motor:\n"
                      f"     python {Path(__file__).name} -c \"{chunks_file}\" -d \"{fallback}\" --provider {other}")
                conn.close()
                raise SystemExit(2)

        rows = [
            (
                item["chunk_id"],
                item.get("source_id", ""),
                item.get("title", ""),
                item.get("url", ""),
                item.get("chunk_index", 0),
                item.get("source", "general"),
                item["text"],
                np.array(vec, dtype=np.float32).tobytes(),
            )
            for item, vec in zip(batch, vectors)
        ]
        cur.executemany(
            """
            INSERT OR REPLACE INTO chunks
            (chunk_id, source_id, title, url, chunk_index, source, text, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        emb.write_index_meta(conn, embedder, note=note)

        processed += len(batch)
        print(f"Progreso: {processed}/{total_pending} ({processed / total_pending * 100:.1f}%)")
        if processed < total_pending and embedder.name == "gemini":
            time.sleep(GEMINI_PAUSE)

    cur.execute("SELECT COUNT(*) FROM chunks")
    total_in_db = cur.fetchone()[0]
    conn.close()
    print(f"Listo. Total chunks en {db_file}: {total_in_db} (motor: {embedder.name})")
    return total_in_db


def main():
    parser = argparse.ArgumentParser(description="Indexador vectorial central para RAG.")
    parser.add_argument("--chunks", "-c", type=str, required=True, help="Ruta de chunks.jsonl a indexar.")
    parser.add_argument("--db", "-d", type=str, required=True, help="Ruta de la base vector_index.db destino.")
    parser.add_argument("--batch", "-b", type=int, default=DEFAULT_BATCH_SIZE, help="Tamaño de lote (defecto: 45).")
    parser.add_argument(
        "--provider", "-p", default="auto", choices=["auto", "gemini", "ollama"],
        help="Motor de embeddings: auto (Gemini y si falla BGE-M3 local), gemini u ollama.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Borra los vectores del índice y lo rehace (necesario para cambiar de motor).",
    )
    args = parser.parse_args()

    index_file(Path(args.chunks), Path(args.db), batch_size=args.batch,
               provider=args.provider, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
