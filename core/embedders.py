#!/usr/bin/env python3
"""
Proveedores de embeddings de RAGnar.

Dos motores, una sola interfaz:

  * ``gemini``  — Gemini Embedding (nube, 768 dim). Requiere API key.
  * ``ollama``  — BGE-M3 servido por Ollama en local (1024 dim). Sin cuotas ni red.

REGLA DE ORO: un índice pertenece a UN solo modelo. Mezclar vectores de 768
dimensiones con otros de 1024 produce similitudes sin sentido, así que el
proveedor se guarda en la tabla ``meta`` de cada ``vector_index.db`` y se
respeta al consultar. Cambiar de motor implica reindexar.

Configuración por variables de entorno:
  RAG_EMBED_PROVIDER   auto|gemini|ollama  (defecto: auto)
  OLLAMA_HOST          (defecto: http://127.0.0.1:11434)
  RAG_OLLAMA_MODEL     (defecto: bge-m3)
"""

import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

GEMINI_MODEL = "models/gemini-embedding-001"
GEMINI_DIM = 768
GEMINI_BATCH = 45
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"

OLLAMA_MODEL = os.environ.get("RAG_OLLAMA_MODEL", "bge-m3")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_BATCH = 16

# Umbral de similitud calibrado por motor sobre el corpus de Michelle Engelmann
# (1.264 chunks, 6 preguntas dentro y 4 fuera del corpus):
#   gemini: dentro 0.630-0.794 · fuera 0.540-0.568 -> corte 0.60
#   bge-m3: dentro 0.589-0.679 · fuera 0.410-0.486 -> corte 0.54
# Un motor con otra distribución de scores necesita el suyo: reutilizar 0.65 con
# BGE-M3 rechazaría preguntas legítimas.
MIN_SCORE_DEFAULTS = {"gemini": 0.60, "ollama": 0.54}

META_PROVIDER = "embedding_provider"
META_MODEL = "embedding_model"
META_DIM = "embedding_dim"


def _short(body: str, limit: int = 160) -> str:
    """Cuerpo de error en una línea, sin JSON crudo ilegible."""
    try:
        payload = json.loads(body)
        message = payload.get("error", {}).get("message") or payload.get("message")
        if message:
            return message[:limit]
    except (ValueError, AttributeError):
        pass
    flat = " ".join(body.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


class EmbeddingError(RuntimeError):
    """Fallo genérico del proveedor de embeddings."""


class ProviderUnavailable(EmbeddingError):
    """El proveedor no puede atender ahora (cuota, saldo, red, servicio caído).

    Marca la diferencia importante: reintentar no sirve, hay que cambiar de motor.
    """


class GeminiEmbedder:
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str = GEMINI_MODEL, dim: int | None = None):
        self.model = model
        # Gemini devuelve su dimensionalidad nativa (3072) si no se le pide otra: nunca dejar None.
        self.dim = int(dim) if dim else GEMINI_DIM
        self._api_key = api_key

    @property
    def api_key(self) -> str | None:
        if self._api_key is None:
            try:
                from config import get_api_key  # import diferido: evita ciclos

                self._api_key = get_api_key()
            except Exception:
                self._api_key = ""
        return self._api_key or None

    def available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "sin API key de Gemini"
        return True, "API key presente"

    def embed(self, texts: list[str], retries: int = 4) -> list[list[float]]:
        if not self.api_key:
            raise ProviderUnavailable("No hay API key de Gemini configurada.")
        url = f"{GEMINI_URL}/{self.model}:batchEmbedContents?key={self.api_key}"
        payload = [
            {"model": self.model, "content": {"parts": [{"text": t}]}, "outputDimensionality": self.dim}
            for t in texts
        ]
        data = json.dumps({"requests": payload}).encode("utf-8")

        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=90) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                vectors = [e["values"] for e in res.get("embeddings", [])]
                if len(vectors) != len(texts):
                    raise EmbeddingError(f"Esperaba {len(texts)} vectores y recibí {len(vectors)}")
                return vectors
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                if e.code == 402:
                    raise ProviderUnavailable(
                        "Gemini devuelve 402: el proyecto no tiene saldo de prepago. "
                        f"Detalle: {_short(body)}"
                    ) from e
                if e.code in (401, 403) or (
                    e.code == 400
                    and any(t in body.lower() for t in ("api key", "api_key", "credential", "authenticat"))
                ):
                    raise ProviderUnavailable(f"Gemini rechaza la clave ({e.code}): {_short(body)}") from e
                if e.code == 429 and attempt < retries - 1:
                    wait = 62
                    print(f"[Aviso] Cuota de Gemini alcanzada. Reintento en {wait}s "
                          f"({attempt + 1}/{retries - 1})...")
                    time.sleep(wait)
                    continue
                if e.code == 429:
                    raise ProviderUnavailable(f"Gemini sin cuota: {_short(body)}") from e
                if e.code in (500, 502, 503) and attempt < retries - 1:
                    time.sleep(2 ** attempt + 2)
                    continue
                raise EmbeddingError(f"Gemini API {e.code}: {_short(body, 300)}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise ProviderUnavailable(f"Sin conexión con Gemini: {e}") from e
        raise EmbeddingError("Gemini agotó los reintentos.")


class OllamaEmbedder:
    name = "ollama"

    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST, dim: int | None = None):
        self.model = model
        self.host = host.rstrip("/")
        self.dim = dim
        self._detected = dim is not None

    def _get(self, path: str, timeout: int = 10):
        with urllib.request.urlopen(f"{self.host}{path}", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def available(self) -> tuple[bool, str]:
        try:
            tags = self._get("/api/tags", timeout=5)
        except Exception as e:
            return False, f"Ollama no responde en {self.host} ({e.__class__.__name__})"
        installed = {m.get("name", "") for m in tags.get("models", [])}
        base = self.model.split(":")[0]
        match = next((n for n in installed if n.split(":")[0] == base), None)
        if not match:
            return False, f"el modelo {self.model} no está en Ollama (instalados: {', '.join(sorted(installed)) or 'ninguno'})"
        self.model = match
        return True, f"{match} listo en {self.host}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), OLLAMA_BATCH):
            chunk = texts[start : start + OLLAMA_BATCH]
            body = json.dumps({"model": self.model, "input": chunk, "truncate": True}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/embed", data=body, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="ignore")[:200]
                raise EmbeddingError(f"Ollama {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise ProviderUnavailable(f"Ollama dejó de responder: {e}") from e
            batch_vectors = res.get("embeddings") or []
            if len(batch_vectors) != len(chunk):
                raise EmbeddingError(f"Ollama devolvió {len(batch_vectors)} vectores para {len(chunk)} textos")
            if not self._detected and batch_vectors:
                self.dim = len(batch_vectors[0])
                self._detected = True
            vectors.extend(batch_vectors)
        return vectors


def build(name: str, **kwargs) -> GeminiEmbedder | OllamaEmbedder:
    """Crea un proveedor por nombre, sin comprobar que esté disponible."""
    normalized = (name or "").strip().lower()
    if normalized == "gemini":
        return GeminiEmbedder(**kwargs)
    if normalized in ("ollama", "bge", "bge-m3"):
        kwargs.pop("api_key", None)
        return OllamaEmbedder(**kwargs)
    raise ValueError(f"Proveedor desconocido: {name!r} (usa gemini u ollama)")


def resolve(preferred: str = "auto", **kwargs) -> tuple[GeminiEmbedder | OllamaEmbedder, str]:
    """Devuelve el proveedor a usar y una nota de por qué se eligió.

    ``auto`` prefiere Gemini y cae a BGE-M3 si Gemini no puede atender o no está
    instalado. Con un nombre explícito se usa ese, y si no está disponible se
    lanza ``ProviderUnavailable`` en lugar de cambiar de motor a escondidas.
    """
    wanted = (preferred or "auto").strip().lower()
    if wanted != "auto":
        embedder = build(wanted, **kwargs)
        ok, detail = embedder.available()
        if not ok:
            raise ProviderUnavailable(f"{embedder.name} no está disponible: {detail}")
        return embedder, detail

    gemini = build("gemini")
    ok, detail = gemini.available()
    if ok:
        return gemini, f"Gemini ({detail})"

    ollama = build("ollama", **{k: v for k, v in kwargs.items() if k == "model"})
    ok_ollama, detail_ollama = ollama.available()
    if ok_ollama:
        return ollama, f"BGE-M3 local ({detail_ollama}); Gemini descartado: {detail}"
    raise ProviderUnavailable(f"Ningún motor de embeddings disponible. Gemini: {detail}. Local: {detail_ollama}")


# --- metadatos del índice -------------------------------------------------

def ensure_meta_table(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()


def read_index_meta(db_path: Path) -> dict:
    """Metadatos guardados en el índice; dict vacío si es un índice antiguo."""
    if not Path(db_path).exists():
        return {}
    try:
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return {}
    meta = {k: v for k, v in rows}
    if META_DIM in meta:
        try:
            meta[META_DIM] = int(meta[META_DIM])
        except (TypeError, ValueError):
            meta.pop(META_DIM, None)
    return meta


def write_index_meta(conn: sqlite3.Connection, embedder, note: str = "") -> None:
    ensure_meta_table(conn)
    values = {
        META_PROVIDER: embedder.name,
        META_MODEL: getattr(embedder, "model", ""),
        META_DIM: str(getattr(embedder, "dim", "") or ""),
        "embedding_updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if note:
        values["embedding_note"] = note
    conn.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", values.items())
    conn.commit()


def default_min_score(provider: str) -> float:
    """Umbral de relevancia recomendado para el motor indicado."""
    return MIN_SCORE_DEFAULTS.get((provider or "").strip().lower(), 0.60)


def meta_provider(meta: dict) -> str:
    """Proveedor de un índice; los índices previos a esta función son de Gemini."""
    return (meta.get(META_PROVIDER) or "gemini").strip().lower()


def index_dim(db_path: Path, meta: dict | None = None) -> int | None:
    meta = read_index_meta(db_path) if meta is None else meta
    if meta.get(META_DIM):
        return meta[META_DIM]
    return GEMINI_DIM if not meta else None


def fallback_db_path(db_path: Path, suffix: str = "bge") -> Path:
    """Índice hermano con vectores locales: ``vector_index.db`` -> ``vector_index.bge.db``."""
    db_path = Path(db_path)
    stem = db_path.name[: -len(db_path.suffix)] if db_path.suffix else db_path.name
    return db_path.with_name(f"{stem}.{suffix}{db_path.suffix}")
