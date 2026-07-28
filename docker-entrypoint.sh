#!/bin/sh
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
export OLLAMA_HOST
FLASK_HOST="${FLASK_HOST:-0.0.0.0}"
FLASK_PORT="${FLASK_PORT:-5000}"
export FLASK_HOST FLASK_PORT

if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
  echo "[entrypoint] Checking ML models (SBERT + Cascade V8)…"
  python scripts/download_models.py || echo "[entrypoint] Model download warning — check logs / volume mounts."
fi

echo "[entrypoint] Attente d Ollama (${OLLAMA_HOST})…"
TRIES=0
until curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; do
  TRIES=$((TRIES + 1))
  if [ "$TRIES" -ge 60 ]; then
    echo "[entrypoint] Ollama indisponible — demarrage Flask quand meme."
    break
  fi
  sleep 2
done

if [ "${SKIP_LLM_PULL:-0}" != "1" ]; then
  echo "[entrypoint] Pull LLM si necessaire…"
  python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from cascade.core import load_config
from ollama_service import ensure_ollama_ready
cfg = load_config()
ensure_ollama_ready(
    cfg, Path("cascade/config.json"),
    install_binary=False, pull_llm=True, quiet=True,
)
print("[entrypoint] LLM pret.")
PY
fi

echo "[entrypoint] Demarrage Flask sur ${FLASK_HOST}:${FLASK_PORT}…"
exec python app.py
