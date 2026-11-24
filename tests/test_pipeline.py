#!/usr/bin/env python3
"""Tests offline del pipeline ETL: chunker y propagación de metadatos.

Cubren los dos fallos que aparecieron al fabricar el primer agente desde un
dataset de Apify: el `.jsonl` mal interpretado y los metadatos del chunk
anidados y perdidos al indexar.

    python -m unittest tests.test_pipeline -v
"""

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(CORE))

import chunker  # noqa: E402
import searcher  # noqa: E402


class TestChunkerJSONL(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_jsonl_con_varios_documentos(self):
        # Un .jsonl empieza por "{": antes se intentaba leer como un JSON único.
        path = self.dir / "docs.jsonl"
        path.write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in [
                {"video_id": "v1", "title": "Uno", "url": "u1", "text": "Texto del primer documento. " * 3},
                {"video_id": "v2", "title": "Dos", "url": "u2", "text": "Texto del segundo documento. " * 3},
            ]),
            encoding="utf-8",
        )
        docs = chunker.load_documents(path)
        self.assertEqual(len(docs), 2)
        self.assertEqual({d["video_id"] for d in docs}, {"v1", "v2"})

    def test_jsonl_renombrado_a_json(self):
        path = self.dir / "docs.json"
        path.write_text(
            json.dumps({"video_id": "v1", "text": "uno"}) + "\n" + json.dumps({"video_id": "v2", "text": "dos"}),
            encoding="utf-8",
        )
        docs = chunker.load_documents(path)
        self.assertEqual(len(docs), 2)

    def test_json_con_lista_envuelta(self):
        path = self.dir / "docs.json"
        path.write_text(json.dumps({"items": [{"a": 1}, {"a": 2}]}), encoding="utf-8")
        self.assertEqual(len(chunker.load_documents(path)), 2)


class TestMetadatosDelChunk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_meta_explicito_se_aplana(self):
        source = self.dir / "entrada.jsonl"
        source.write_text(json.dumps({
            "video_id": "v1",
            "title": "Título",
            "url": "https://ejemplo",
            "text": "Contenido " * 40,
            "meta": {"perfil": "Perfil X", "estado_fuente": "DISPONIBLE"},
            "modalidad": "video",
        }, ensure_ascii=False), encoding="utf-8")

        out = self.dir / "chunks.jsonl"
        chunker.build_chunks(source, out, source_label="youtube")
        chunk = json.loads(out.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(chunk["source_id"], "v1")
        self.assertEqual(chunk["source"], "youtube")
        self.assertEqual(chunk["meta"]["perfil"], "Perfil X")
        self.assertEqual(chunk["meta"]["modalidad"], "video")
        self.assertNotIn("meta", chunk["meta"])  # nada de {"meta": {"meta": ...}}

    def test_campos_canonicos_presentes(self):
        source = self.dir / "entrada.json"
        source.write_text(json.dumps([{"id": "d1", "title": "T", "url": "u", "text": "x " * 600}]), encoding="utf-8")
        out = self.dir / "chunks.jsonl"
        chunker.build_chunks(source, out)
        for line in out.read_text(encoding="utf-8").splitlines():
            chunk = json.loads(line)
            for key in ("chunk_id", "source_id", "title", "url", "chunk_index", "source", "text"):
                self.assertIn(key, chunk)


class TestMetaEnElIndice(unittest.TestCase):
    def test_el_buscador_devuelve_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "idx.db"
            conn = sqlite3.connect(db)
            conn.execute(
                "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, title TEXT, url TEXT, text TEXT, "
                "embedding BLOB, meta TEXT)"
            )
            import numpy as np

            conn.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                ("a:000", "t", "u", "texto", np.array([1.0, 0.0], dtype=np.float32).tobytes(),
                 json.dumps({"perfil": "Perfil X", "estado_fuente": "DISPONIBLE"})),
            )
            conn.commit()
            conn.close()

            results, _ = searcher._load_and_rank(db, np.array([1.0, 0.0], dtype=np.float32), 1, 0.0)
            self.assertEqual(results[0]["meta"]["perfil"], "Perfil X")

    def test_indice_sin_columna_meta_sigue_funcionando(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "idx.db"
            conn = sqlite3.connect(db)
            conn.execute(
                "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, title TEXT, url TEXT, text TEXT, embedding BLOB)"
            )
            import numpy as np

            conn.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?)",
                ("a:000", "t", "u", "texto", np.array([1.0, 0.0], dtype=np.float32).tobytes()),
            )
            conn.commit()
            conn.close()

            results, _ = searcher._load_and_rank(db, np.array([1.0, 0.0], dtype=np.float32), 1, 0.0)
            self.assertEqual(len(results), 1)
            self.assertNotIn("meta", results[0])


if __name__ == "__main__":
    unittest.main()
