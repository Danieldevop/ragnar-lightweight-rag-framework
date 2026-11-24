#!/usr/bin/env python3
"""
Buscador Semántico Central para RAG.

Calcula el embedding de la consulta con el mismo motor que construyó el índice
y busca por similitud coseno sobre cualquier vector_index.db en milisegundos.

Respaldo automático: si el motor del índice no puede atender (típico: Gemini
devuelve 402 o se queda sin cuota) y existe el índice hermano ``*.bge.db``
construido con BGE-M3 local, la consulta se responde con ese índice. Nunca se
mezclan vectores de modelos distintos en la misma búsqueda.

  python searcher.py -d vector_index.db "pregunta" --top 3 --min-score 0.65
  python searcher.py -d vector_index.db "pregunta" --no-fallback
  python searcher.py -d vector_index.db "pregunta" --fallback-db otra.bge.db
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedders as emb  # noqa: E402


def _embed_query(query: str, db_path: Path, provider: str = "auto") -> tuple[np.ndarray, str]:
    """Vectoriza la consulta con el motor del índice. Devuelve (vector, motor)."""
    meta = emb.read_index_meta(db_path)
    stored = emb.meta_provider(meta) if meta else "gemini"
    wanted = (provider or "auto").strip().lower()

    if wanted != "auto" and wanted != stored:
        raise SystemExit(
            f"⚠️  El índice {Path(db_path).name} usa el motor '{stored}', no '{wanted}'.\n"
            f"    Usa --provider {stored} o consulta el índice construido con {wanted}."
        )

    embedder = emb.build(stored, dim=meta.get(emb.META_DIM))
    ok, detail = embedder.available()
    if not ok:
        raise emb.ProviderUnavailable(f"{stored} no disponible: {detail}")

    vector = np.array(embedder.embed([query])[0], dtype=np.float32)
    return vector, stored


def _load_and_rank(db_path: Path, q_vec: np.ndarray, top_k: int, min_score: float):
    conn = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        extra = ", meta" if "meta" in columns else ", NULL"
        rows = conn.execute(f"SELECT chunk_id, title, url, text, embedding{extra} FROM chunks").fetchall()
    finally:
        conn.close()
    if not rows:
        return [], 0.0

    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm

    embs = np.array([np.frombuffer(r[4], dtype=np.float32) for r in rows])
    if embs.shape[1] != q_vec.shape[0]:
        raise SystemExit(
            f"❌ Dimensiones incompatibles en {Path(db_path).name}: el índice guarda {embs.shape[1]} y la "
            f"consulta generó {q_vec.shape[0]}. Reindexa ese índice con --rebuild y el motor correcto."
        )

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    scores = np.dot(embs / norms, q_vec)
    order = np.argsort(scores)[::-1]

    results = []
    for idx in order:
        score = float(scores[idx])
        if score < min_score:
            break  # el orden es descendente: nada mejor queda por venir
        record = {
            "chunk_id": rows[idx][0],
            "title": rows[idx][1],
            "url": rows[idx][2],
            "score": score,
            "text": rows[idx][3],
        }
        if len(rows[idx]) > 5 and rows[idx][5]:
            try:
                record["meta"] = json.loads(rows[idx][5])
            except (ValueError, TypeError):
                pass
        results.append(record)
        if len(results) >= top_k:
            break
    return results, float(scores[order[0]])


def search_vector_db(
    query: str,
    db_path: Path,
    top_k: int = 4,
    min_score: float | None = None,
    provider: str = "auto",
    fallback: bool = True,
    fallback_db: Path | None = None,
    diagnostics: dict | None = None,
):
    """Busca por similitud coseno. Devuelve (resultados, mejor_puntaje).

    ``diagnostics``, si se pasa un dict, se rellena con el motor usado y si hubo
    respaldo, para que el llamador pueda avisar al usuario.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"No existe la base de vectores: {db_path}")

    diag = diagnostics if diagnostics is not None else {}
    diag.setdefault("fallback", None)

    try:
        q_vec, used = _embed_query(query, db_path, provider)
    except emb.ProviderUnavailable as e:
        if not fallback:
            raise
        alternate = Path(fallback_db) if fallback_db else emb.fallback_db_path(db_path)
        if not alternate.exists():
            raise SystemExit(
                f"❌ {e}\n"
                f"   No hay índice de respaldo en {alternate}.\n"
                f"   Créalo una vez con: python indexer.py -c chunks.jsonl -d \"{alternate}\" --provider ollama"
            ) from e
        print(f"⚠️  {e}\n   → Respondiendo con el índice local {alternate.name} (BGE-M3).", file=sys.stderr)
        q_vec, used = _embed_query(query, alternate, "auto")
        diag["fallback"] = {"from": emb.meta_provider(emb.read_index_meta(db_path)), "db": str(alternate)}
        db_path = alternate

    diag["provider"] = used
    diag["db"] = str(db_path)
    threshold = emb.default_min_score(used) if min_score is None else min_score
    diag["min_score"] = threshold
    results, best = _load_and_rank(db_path, q_vec, top_k, threshold)
    return results, best


def main():
    parser = argparse.ArgumentParser(description="Buscador semántico central para RAG.")
    parser.add_argument("--db", "-d", type=str, required=True, help="Ruta de vector_index.db a consultar.")
    parser.add_argument("query", type=str, help="Pregunta o concepto a buscar.")
    parser.add_argument("--top", "-k", type=int, default=4, help="Número de resultados a devolver.")
    parser.add_argument(
        "--min-score", "-m", type=float, default=None,
        help="Similitud mínima (0-1). Por defecto se usa el umbral calibrado del motor "
             "(Gemini 0.60, BGE-M3 0.54); pasar 0 desactiva el filtro.",
    )
    parser.add_argument(
        "--provider", "-p", default="auto", choices=["auto", "gemini", "ollama"],
        help="Motor de embeddings de la consulta. Por defecto, el que grabó el índice.",
    )
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="No usar el índice local *.bge.db si el motor principal falla.",
    )
    parser.add_argument(
        "--fallback-db", type=str, default=None,
        help="Índice de respaldo explícito (por defecto: <db>.bge.db hermano).",
    )
    parser.add_argument("--json", action="store_true", help="Salida en JSON crudo.")
    args = parser.parse_args()

    diag: dict = {}
    try:
        results, best = search_vector_db(
            args.query,
            Path(args.db),
            top_k=args.top,
            min_score=args.min_score,
            provider=args.provider,
            fallback=not args.no_fallback,
            fallback_db=Path(args.fallback_db) if args.fallback_db else None,
            diagnostics=diag,
        )
    except emb.ProviderUnavailable as e:
        print(f"\n❌ {e}", file=sys.stderr)
        print(f"   Sin respaldo local: crea el índice con --provider ollama o pasa --fallback-db.", file=sys.stderr)
        raise SystemExit(1) from e
    except emb.EmbeddingError as e:
        print(f"\n❌ Error de embeddings: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    if args.json:
        print(json.dumps({"results": results, "best_score": best, "engine": diag}, ensure_ascii=False, indent=2))
        return

    if not results:
        umbral = diag.get("min_score", 0.0)
        motor = diag.get("provider")
        respaldo = " · RESPALDO LOCAL" if diag.get("fallback") else ""
        mejor = f" (el mejor fragmento quedó en {best:.4f})" if umbral > 0 else ""
        print(f"\n⚠️  Sin resultados para: \"{args.query}\"")
        print(f"    Motor: {motor} · min-score {umbral:.2f}{respaldo}")
        print(f"    Ningún fragmento llega a ese umbral{mejor}")
        print("    → Este tema no está cubierto en el corpus; responde sin citar fuentes.")
        return

    print(f"\n🔎 Resultados para: \"{args.query}\" (Top {len(results)})\n" + "=" * 60)
    print(f"    Motor: {diag.get('provider')} · min-score {diag.get('min_score', 0):.2f}"
          + (" · RESPALDO LOCAL" if diag.get("fallback") else ""))
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['title']} (Similitud: {r['score']:.4f})")
        print(f"    URL: {r['url']} | ID: {r['chunk_id']}")
        print(f"    Texto:\n    {r['text']}\n")


if __name__ == "__main__":
    main()
