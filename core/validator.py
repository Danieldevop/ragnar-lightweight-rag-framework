#!/usr/bin/env python3
"""
Validador de chunks.jsonl contra el esquema estándar de RAG.
Verifica campos requeridos, tipos, formato de chunk_id e invariantes.
"""

import sys
import json
import argparse
from pathlib import Path

REQUIRED_KEYS = ["chunk_id", "source_id", "title", "url", "chunk_index", "source", "text"]

def validate_file(file_path: Path):
    if not file_path.exists():
        print(f"[ERROR] No existe el archivo: {file_path}")
        return False

    errors = []
    seen_ids = set()
    doc_indices = {}

    with open(file_path, "r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                errors.append(f"Línea {line_num}: JSON inválido ({e})")
                continue

            for k in REQUIRED_KEYS:
                if k not in rec:
                    errors.append(f"Línea {line_num}: Falta campo obligatorio '{k}'")

            cid = rec.get("chunk_id", "")
            sid = rec.get("source_id", "")
            cidx = rec.get("chunk_index")
            text = rec.get("text", "")

            if cid in seen_ids:
                errors.append(f"Línea {line_num}: chunk_id duplicado '{cid}'")
            seen_ids.add(cid)

            if ":" in cid:
                expected_prefix = cid.split(":", 1)[0]
                if expected_prefix != str(sid):
                    errors.append(f"Línea {line_num}: chunk_id '{cid}' no coincide con source_id '{sid}'")
            else:
                errors.append(f"Línea {line_num}: chunk_id '{cid}' no tiene formato source_id:NNN")

            if not isinstance(cidx, int) or cidx < 0:
                errors.append(f"Línea {line_num}: chunk_index debe ser entero >= 0")
            else:
                expected_idx = doc_indices.get(sid, 0)
                if cidx != expected_idx:
                    errors.append(f"Línea {line_num}: chunk_index {cidx} no secuencial (esperado {expected_idx}) para doc '{sid}'")
                doc_indices[sid] = cidx + 1

            if not isinstance(text, str) or not text.strip():
                errors.append(f"Línea {line_num}: 'text' vacío o no es string")

    if errors:
        print(f"❌ Validación fallida con {len(errors)} incidencias:")
        for err in errors[:15]:
            print(f"  - {err}")
        if len(errors) > 15:
            print(f"  ... y {len(errors) - 15} errores más.")
        return False

    total_chunks = len(seen_ids)
    total_docs = len(doc_indices)
    print(f"✅ Validación exitosa: {total_chunks} chunks de {total_docs} documentos cumplen el estándar.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Valida un archivo chunks.jsonl.")
    parser.add_argument("file", type=str, help="Ruta de chunks.jsonl a validar.")
    args = parser.parse_args()

    ok = validate_file(Path(args.file))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
