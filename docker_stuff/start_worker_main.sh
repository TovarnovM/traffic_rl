#!/usr/bin/env bash
set -euo pipefail

export RAY_EXTRA_RESOURCES_JSON="$(
  EXTRA_RESOURCES_JSON="${RAY_EXTRA_RESOURCES_JSON:-}" python - <<'PY'
import json
import os

raw = os.environ.get("EXTRA_RESOURCES_JSON", "").strip()
resources = {}

if raw:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("RAY_EXTRA_RESOURCES_JSON must be a JSON object")
    resources.update(parsed)

resources["main"] = 1

print(json.dumps(resources, sort_keys=True, separators=(",", ":")))
PY
)"

exec /ray_bootstrap/start_worker.sh