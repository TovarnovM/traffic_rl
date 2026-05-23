#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

NODE_IP="$(python - <<'PY'
import socket

hostname = socket.gethostname()
ips = []
for fam, _, _, _, sa in socket.getaddrinfo(hostname, None):
    if fam == socket.AF_INET:
        ip = sa[0]
        if not ip.startswith("127."):
            ips.append(ip)

print(ips[0] if ips else "")
PY
)"

[ -n "${NODE_IP}" ] || {
  echo "[ray-head] cannot determine non-loopback IPv4"
  exit 1
}

exec ray start \
  --head \
  --node-ip-address="${NODE_IP}" \
  --port="${RAY_HEAD_PORT:-6379}" \
  --ray-client-server-port="${RAY_CLIENT_PORT:-8900}" \
  --include-dashboard=true \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT:-8265}" \
  --disable-usage-stats \
  --object-store-memory="${RAY_OBJECT_STORE_MEMORY:-8589934592}" \
  --num-cpus="${RAY_HEAD_NUM_CPUS:-0}" \
  --num-gpus="${RAY_HEAD_NUM_GPUS:-0}" \
  ${RAY_PASSWORD:+--redis-password="${RAY_PASSWORD}"} \
  --block