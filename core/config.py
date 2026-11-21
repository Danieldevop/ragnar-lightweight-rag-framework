#!/usr/bin/env python3
"""
Configuración compartida del framework RAG.

Resuelve la API key de Gemini sin credenciales incrustadas en el código.
Orden de búsqueda:
  1. Variable de entorno GEMINI_API_KEY
  2. Archivo indicado en GEMINI_KEY_FILE
  3. Archivo .gemini_api_key junto al script, al directorio actual o al db
  4. Vault local de OpenBot (%APPDATA%/OpenBot/agent-api-keys/.vault/*/GEMINI_API_KEY)
"""

import os
from pathlib import Path

KEY_FILENAME = "GEMINI_API_KEY"
ENV_FILE_NAME = ".gemini_api_key"


def _read(path: Path) -> str | None:
    try:
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            return value or None
    except OSError:
        return None
    return None


def _candidate_dirs(hint: Path | None = None) -> list[Path]:
    dirs = [Path(__file__).resolve().parent, Path.cwd()]
    if hint is not None:
        dirs.append(Path(hint).resolve().parent)
    return dirs


def get_api_key(hint: Path | None = None) -> str:
    """Devuelve la API key de Gemini o lanza RuntimeError si no existe."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(var)
        if value and value.strip():
            return value.strip()

    env_file = os.environ.get("GEMINI_KEY_FILE")
    if env_file:
        value = _read(Path(env_file))
        if value:
            return value

    for directory in _candidate_dirs(hint):
        value = _read(directory / ENV_FILE_NAME)
        if value:
            return value

    appdata = os.environ.get("APPDATA")
    if appdata:
        vault = Path(appdata) / "OpenBot" / "agent-api-keys" / ".vault"
        for candidate in sorted(vault.glob(f"*/{KEY_FILENAME}")):
            value = _read(candidate)
            if value:
                return value

    raise RuntimeError(
        "No se encontró la API key de Gemini. Define GEMINI_API_KEY, "
        "GEMINI_KEY_FILE o crea un archivo .gemini_api_key junto al script."
    )
