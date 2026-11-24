#!/usr/bin/env python3
"""Tests offline del núcleo de RAGnar: sin red, sin API key, sin Ollama.

    python -m unittest discover -s tests -v

Cubren las piezas que más fácil se rompen: los metadatos del índice, los
umbrales por motor y el rechazo a mezclar vectores de modelos distintos.
"""

import sqlite3
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

CORE = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(CORE))

import embedders as emb  # noqa: E402
import searcher  # noqa: E402


class FakeEmbedder:
    """Proveedor mínimo para probar metadatos sin llamar a nadie."""

    def __init__(self, name="gemini", model="models/gemini-embedding-001", dim=768):
        self.name = name
        self.model = model
        self.dim = dim


class TestMeta(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "vector_index.db"
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY)")
        emb.ensure_meta_table(conn)
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        conn = sqlite3.connect(self.db)
        emb.write_index_meta(conn, FakeEmbedder(name="ollama", model="bge-m3:latest", dim=1024), note="local")
        conn.close()

        meta = emb.read_index_meta(self.db)
        self.assertEqual(meta[emb.META_PROVIDER], "ollama")
        self.assertEqual(meta[emb.META_MODEL], "bge-m3:latest")
        self.assertEqual(meta[emb.META_DIM], 1024)  # se devuelve como int
        self.assertEqual(meta["embedding_note"], "local")

    def test_indice_antiguo_sin_meta_es_gemini(self):
        self.assertEqual(emb.read_index_meta(self.db), {})
        self.assertEqual(emb.meta_provider({}), "gemini")

    def test_db_inexistente_no_revienta(self):
        self.assertEqual(emb.read_index_meta(Path(self.tmp.name) / "no-existe.db"), {})


class TestUmbrales(unittest.TestCase):
    def test_umbral_por_motor(self):
        self.assertAlmostEqual(emb.default_min_score("gemini"), 0.60)
        self.assertAlmostEqual(emb.default_min_score("ollama"), 0.54)

    def test_motor_desconocido_cae_al_default(self):
        self.assertAlmostEqual(emb.default_min_score("otro"), 0.60)


class TestEmbedders(unittest.TestCase):
    def test_gemini_nunca_deja_dim_en_none(self):
        # Gemini devuelve 3072 dims si no se le pide una dimensionalidad concreta.
        self.assertEqual(emb.GeminiEmbedder(dim=None).dim, 768)
        self.assertEqual(emb.GeminiEmbedder(dim=1024).dim, 1024)

    def test_build_por_nombre(self):
        self.assertIsInstance(emb.build("gemini"), emb.GeminiEmbedder)
        self.assertIsInstance(emb.build("ollama"), emb.OllamaEmbedder)
        self.assertIsInstance(emb.build("bge-m3"), emb.OllamaEmbedder)
        with self.assertRaises(ValueError):
            emb.build("inventado")

    def test_gemini_402_es_indisponible_no_reintentable(self):
        error = urllib.error.HTTPError(
            "http://x", 402, "Payment Required", {}, mock.MagicMock(read=lambda *_: b'{"error":{"message":"sin saldo"}}')
        )
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(emb.ProviderUnavailable):
                emb.GeminiEmbedder(api_key="k", dim=768).embed(["hola"])

    def test_gemini_clave_invalida_es_indisponible(self):
        body = b'{"error":{"code":400,"message":"API key not valid. Please pass a valid API key."}}'
        error = urllib.error.HTTPError("http://x", 400, "Bad Request", {}, mock.MagicMock(read=lambda *_: body))
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(emb.ProviderUnavailable):
                emb.GeminiEmbedder(api_key="k", dim=768).embed(["hola"])

    def test_gemini_400_generico_no_se_confunde_con_caida(self):
        body = b'{"error":{"code":400,"message":"Request payload size exceeds the limit"}}'
        error = urllib.error.HTTPError("http://x", 400, "Bad Request", {}, mock.MagicMock(read=lambda *_: body))
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(emb.EmbeddingError) as ctx:
                emb.GeminiEmbedder(api_key="k", dim=768).embed(["hola"])
            self.assertNotIsInstance(ctx.exception, emb.ProviderUnavailable)

    def test_ollama_sin_servidor_no_revienta(self):
        ok, detail = emb.OllamaEmbedder(host="http://127.0.0.1:9", model="bge-m3").available()
        self.assertFalse(ok)
        self.assertIn("no responde", detail)

    def test_short_extrae_el_mensaje_del_json(self):
        self.assertEqual(emb._short('{"error":{"message":"cuota agotada"}}'), "cuota agotada")


class TestRutas(unittest.TestCase):
    def test_indice_hermano_local(self):
        self.assertEqual(
            emb.fallback_db_path(Path("proj/fuente/vector_index.db")),
            Path("proj/fuente/vector_index.bge.db"),
        )


class TestDimensiones(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "idx.db"
        conn = sqlite3.connect(self.db)
        conn.execute(
            "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, title TEXT, url TEXT, text TEXT, embedding BLOB)"
        )
        import numpy as np

        conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?)",
                     ("a:000", "t", "u", "texto", np.zeros(4, dtype=np.float32).tobytes()))
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_dimensiones_incompatibles_falla_claro(self):
        import numpy as np

        with self.assertRaises(SystemExit) as ctx:
            searcher._load_and_rank(self.db, np.zeros(8, dtype=np.float32), top_k=1, min_score=0.0)
        self.assertIn("Dimensiones incompatibles", str(ctx.exception))

    def test_ranking_ordena_por_similitud(self):
        import numpy as np

        results, best = searcher._load_and_rank(self.db, np.zeros(4, dtype=np.float32), top_k=1, min_score=0.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(best, 0.0)  # vector nulo: similitud 0, sin NaN


class TestIndexerNoMezcla(unittest.TestCase):
    def test_rechaza_cambiar_de_motor_sin_rebuild(self):
        import indexer

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "idx.db"
            meta = {emb.META_PROVIDER: "gemini"}
            with self.assertRaises(SystemExit) as ctx:
                indexer._select_embedder(db, meta, "ollama")
            self.assertIn("reindexar", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
