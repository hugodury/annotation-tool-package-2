#!/bin/bash
# ISIALAB Annotation Interface — Mac / Linux launcher
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_CMD=""
for cmd in python3 python py; do
    if command -v "$cmd" &>/dev/null && $cmd --version &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.9+ required. https://www.python.org/downloads/"
    exit 1
fi

exec "$PYTHON_CMD" scripts/setup.py --start "$@"
