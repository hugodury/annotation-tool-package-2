#!/bin/bash
# ISIALAB Annotation Interface — Mac / Linux launcher
set -euo pipefail
cd "$(dirname "$0")"

install_python_linux_mac() {
    echo "Python 3.9+ introuvable — installation automatique..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
    elif command -v brew &>/dev/null; then
        brew install python@3.12
    else
        echo "Installez Python 3.9+ : https://www.python.org/downloads/"
        exit 1
    fi
}

PYTHON_CMD=""
for cmd in python3 python py; do
    if command -v "$cmd" &>/dev/null; then
        if $cmd -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)" 2>/dev/null; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    install_python_linux_mac
    PYTHON_CMD="python3"
fi

exec "$PYTHON_CMD" scripts/setup.py --start "$@"
