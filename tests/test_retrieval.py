#!/usr/bin/env python3
"""Tests de recuperación (integración): preguntas de oro contra un corpus real.

Necesitan red o Ollama, así que se saltan salvo que se pidan explícitamente:

    RAG_E2E=1 python -m unittest tests.test_retrieval -v

Corpus por defecto: el de Michelle Engelmann si existe. Para otro:

    RAG_CORPUS="ruta/al/vector_index.db" RAG_E2E=1 python -m unittest tests.test_retrieval -v

Las preguntas de oro son la red de seguridad del retrieval: si un cambio en el
motor, el troceado o el umbral deja de encontrar lo que antes encontraba, aquí
se ve. Cada motor tiene su umbral calibrado, así que se prueban los dos.
"""

import os
import sys
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(CORE))

import embedders as emb  # noqa: E402
import searcher  # noqa: E402

CORPUS = Path(
    os.environ.get(
        "RAG_CORPUS",
        "C:/Users/USER/AI playground/rag/michelle-engelmann/youtube/vector_index.db",
    )
)

ENABLED = os.environ.get("RAG_E2E") == "1" and CORPUS.exists()

DENTRO = [
    ("¿cómo negociar un aumento de sueldo?", "aumento"),
    ("¿qué preguntas hacer en una entrevista de trabajo?", "entrevista"),
    ("¿cómo mejorar mi hoja de vida?", "cv"),
]

FUERA = [
    "¿cuál es la receta de los tamales?",
    "¿cómo configuro un servidor Linux?",
]


@unittest.skipUnless(ENABLED, "RAG_E2E=1 y un corpus real son necesarios")
class TestRecuperacion(unittest.TestCase):
    def _indices(self):
        """El índice principal y su gemelo local, los que existan."""
        candidates = [CORPUS, emb.fallback_db_path(CORPUS)]
        return [db for db in candidates if db.exists()]

    def test_preguntas_del_corpus_devuelven_fragmento(self):
        for db in self._indices():
            motor = emb.meta_provider(emb.read_index_meta(db))
            umbral = emb.default_min_score(motor)
            for pregunta, esperado in DENTRO:
                with self.subTest(db=db.name, pregunta=pregunta):
                    results, best = searcher.search_vector_db(
                        pregunta, db, top_k=3, fallback=False
                    )
                    self.assertTrue(results, f"{motor} no devolvió nada para {pregunta!r} (mejor {best:.3f})")
                    self.assertGreaterEqual(best, umbral, f"{motor}: {best:.3f} por debajo de {umbral}")
                    titulos = " ".join(r["title"].lower() for r in results)
                    self.assertIn(esperado, titulos, f"esperaba un fragmento sobre {esperado!r}, llegó: {titulos[:120]}")

    def test_preguntas_fuera_del_corpus_no_devuelven_nada(self):
        for db in self._indices():
            motor = emb.meta_provider(emb.read_index_meta(db))
            for pregunta in FUERA:
                with self.subTest(db=db.name, pregunta=pregunta):
                    results, _ = searcher.search_vector_db(
                        pregunta, db, top_k=3, fallback=False
                    )
                    self.assertEqual(
                        results, [], f"{motor} respondió a una pregunta fuera de tema: {pregunta!r}"
                    )

    def test_umbral_explicito_se_respeta(self):
        db = CORPUS if CORPUS.exists() else None
        self.assertIsNotNone(db)
        results, _ = searcher.search_vector_db(
            "¿cómo negociar un aumento de sueldo?", db, top_k=1, min_score=0.99, fallback=False
        )
        self.assertEqual(results, [], "con min-score 0.99 no debería pasar nada")


if __name__ == "__main__":
    unittest.main()
