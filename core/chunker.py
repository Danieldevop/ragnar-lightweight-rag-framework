#!/usr/bin/env python3
"""
Módulo ETL Universal y Agnóstico para RAG.
Soporta JSON, JSONL, CSV, TXT, Markdown (.md) y carpetas completas de documentos.
Limpia, normaliza y trocea texto en chunks.jsonl estándar con los 7 campos canónicos:
chunk_id, source_id, title, url, chunk_index, source, text (y meta opcional).
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path

TARGET_CHARS = 1000
OVERLAP_CHARS = 180
MAX_SENTENCE = 500
MIN_TAIL = 150

BOUNDARY_RE = re.compile(r"[.!?…](?=\s|\n|$)")

# Campos comunes donde suelen venir los textos, títulos y enlaces
TEXT_CANDIDATES = ["text", "content", "body", "raw_text", "article", "transcript", "document", "message", "description"]
TITLE_CANDIDATES = ["title", "name", "subject", "header", "headline", "topic"]
ID_CANDIDATES = ["source_id", "id", "doc_id", "key", "uid", "video_id"]
URL_CANDIDATES = ["url", "link", "webpage_url", "uri", "source_url", "file_path"]

def split_into_chunks(text: str, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS):
    """Trocea cualquier texto respetando límites de oraciones y garantizando solapamiento."""
    text = text.strip()
    n = len(text)
    if not text:
        return []
    if n <= target:
        return [text]

    chunks = []
    start = 0
    while start < n:
        end = min(start + target, n)
        if end < n:
            m = BOUNDARY_RE.search(text, end - 1, min(n, end + MAX_SENTENCE))
            if m:
                end = m.end()
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break

        nxt = max(end - overlap, start + 1)
        ws = text.find(" ", nxt, min(n, nxt + 50))
        if ws != -1:
            nxt = ws + 1
        start = nxt

    if len(chunks) > 1 and len(chunks[-1]) < MIN_TAIL:
        chunks[-2] = chunks[-2] + " " + chunks.pop()

    return chunks

def extract_field(item: dict, candidates: list, default=None):
    """Busca de forma agnóstica el primer campo que exista de una lista de candidatos."""
    for k in candidates:
        if k in item and item[k]:
            return item[k]
    return default

def load_from_json(path: Path):
    """Lee JSON o JSONL agnósticamente (detecta listas, diccionarios envueltos o líneas)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    if not content:
        return []

    if content.startswith("["):
        return json.loads(content)
    elif content.startswith("{"):
        parsed = json.loads(content)
        # Si el dict envuelve una lista (ej: {"data": [...]}, {"documents": [...]}, etc.)
        for key in ["items", "results", "data", "documents", "records", "pages", "articles", "rows"]:
            if isinstance(parsed.get(key), list):
                return parsed[key]
        # Si no, busca cualquier valor que sea lista de diccionarios
        for v in parsed.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return [parsed]
    else:
        # JSONL
        docs = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                docs.append(json.loads(line))
        return docs

def load_from_csv(path: Path):
    """Lee un archivo CSV o TSV como lista de diccionarios."""
    docs = []
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            docs.append(dict(row))
    return docs

def load_from_text_or_md(path: Path):
    """Lee un archivo .md o .txt extrayendo título y contenido."""
    raw = path.read_text(encoding="utf-8-sig")
    title = path.stem
    text = raw

    # Intentar extraer título de frontmatter o primer encabezado H1
    h1_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    if h1_match:
        title = h1_match.group(1).strip()

    return [{
        "source_id": path.stem,
        "title": title,
        "url": str(path.as_posix()),
        "text": text
    }]

def load_documents(input_path: Path):
    """Carga documentos agnósticamente desde archivo o directorio."""
    if not input_path.exists():
        raise FileNotFoundError(f"No existe la ruta: {input_path}")

    # Si es una carpeta entera de archivos (.md, .txt, .json, .csv)
    if input_path.is_dir():
        all_docs = []
        for file in sorted(input_path.rglob("*")):
            if file.is_file() and not file.name.startswith("."):
                suf = file.suffix.lower()
                if suf in (".md", ".txt"):
                    all_docs.extend(load_from_text_or_md(file))
                elif suf in (".json", ".jsonl"):
                    all_docs.extend(load_from_json(file))
                elif suf in (".csv", ".tsv"):
                    all_docs.extend(load_from_csv(file))
        return all_docs

    # Si es un archivo individual
    suf = input_path.suffix.lower()
    if suf in (".json", ".jsonl"):
        return load_from_json(input_path)
    elif suf in (".csv", ".tsv"):
        return load_from_csv(input_path)
    elif suf in (".md", ".txt"):
        return load_from_text_or_md(input_path)
    else:
        # Intento de lectura en texto plano
        return load_from_text_or_md(input_path)

def build_chunks(input_path: Path, output_file: Path, source_label: str = "general",
                 target_chars: int = TARGET_CHARS, overlap_chars: int = OVERLAP_CHARS):
    """Proceso ETL agnóstico: carga, normaliza y escribe chunks.jsonl."""
    raw_docs = load_documents(input_path)
    if not raw_docs:
        print(f"[Aviso] No se encontraron documentos válidos en {input_path}")
        return 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for i, doc in enumerate(raw_docs, 1):
        if not isinstance(doc, dict):
            continue

        sid = extract_field(doc, ID_CANDIDATES, default=f"doc_{i:03d}")
        # Sanitizar source_id para no tener dos puntos ':' ni espacios
        sid = re.sub(r"[:\s]+", "_", str(sid).strip())

        title = extract_field(doc, TITLE_CANDIDATES, default=f"Documento {i}")
        url = extract_field(doc, URL_CANDIDATES, default="")
        text = extract_field(doc, TEXT_CANDIDATES, default="")

        if not str(text).strip():
            continue

        # Metadatos extra libres
        known_keys = set(TEXT_CANDIDATES + TITLE_CANDIDATES + ID_CANDIDATES + URL_CANDIDATES)
        extra_meta = {k: v for k, v in doc.items() if k not in known_keys}

        # Trocear texto
        pieces = split_into_chunks(str(text), target=target_chars, overlap=overlap_chars)

        for cidx, chunk_text in enumerate(pieces):
            rec = {
                "chunk_id": f"{sid}:{cidx:03d}",
                "source_id": sid,
                "title": str(title).strip(),
                "url": str(url).strip(),
                "chunk_index": cidx,
                "source": str(source_label).strip(),
                "text": chunk_text
            }
            if extra_meta:
                rec["meta"] = extra_meta
            all_chunks.append(rec)

    with open(output_file, "w", encoding="utf-8") as out:
        for ch in all_chunks:
            out.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"ETL Agnóstico completado: {len(all_chunks)} chunks generados de {len(raw_docs)} documentos en {output_file}")
    return len(all_chunks)

def main():
    parser = argparse.ArgumentParser(description="ETL Agnóstico para generar chunks.jsonl para RAG.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Archivo (.json, .jsonl, .csv, .md, .txt) o carpeta de documentos.")
    parser.add_argument("--output", "-o", type=str, default="chunks.jsonl", help="Ruta de chunks.jsonl destino.")
    parser.add_argument("--source", "-s", type=str, default="general", help="Etiqueta del origen (ej: youtube, web, pdf, libro, docs).")
    parser.add_argument("--target", "-t", type=int, default=TARGET_CHARS, help="Caracteres objetivo por chunk (defecto: 1000).")
    parser.add_argument("--overlap", type=int, default=OVERLAP_CHARS, help="Caracteres de solapamiento (defecto: 180).")
    args = parser.parse_args()

    build_chunks(
        input_path=Path(args.input),
        output_file=Path(args.output),
        source_label=args.source,
        target_chars=args.target,
        overlap_chars=args.overlap
    )

if __name__ == "__main__":
    main()
