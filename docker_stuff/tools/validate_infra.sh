#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "$ROOT_DIR"

docker compose -f docker-compose.dev.yml config >/dev/null

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -c "import snfs_traffic; print(snfs_traffic.__version__)"
