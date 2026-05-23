#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

HEAD_HOST="${RAY_HEAD_HOST:-tasks.ray_head}"
HEAD_PORT="${RAY_HEAD_PORT:-6379}"
HEAD_URL="${RAY_HEAD_URL:-${HEAD_HOST}:${HEAD_PORT}}"

HEAD_IP="$(getent hosts "${HEAD_HOST}" | awk 'NR==1{print $1}')"
[ -n "${HEAD_IP}" ] || {
  echo "[ray-worker] cannot resolve head host: ${HEAD_HOST}"
  exit 1
}

NODE_IP="$(HEAD_IP="${HEAD_IP}" HEAD_PORT="${HEAD_PORT}" python - <<'PY'
import os
import socket

head_ip = os.environ["HEAD_IP"]
head_port = int(os.environ["HEAD_PORT"])

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.connect((head_ip, head_port))
print(sock.getsockname()[0])
sock.close()
PY
)"
[ -n "${NODE_IP}" ] || {
  echo "[ray-worker] cannot determine node IPv4"
  exit 1
}

MEM_GB="$(awk '/MemTotal/ { printf "%.0f\n", $2/1024/1024 }' /proc/meminfo)"
MEM_GB="${MEM_GB_OVERRIDE:-${MEM_GB}}"

CPU_COUNT="${RAY_WORKER_NUM_CPUS:-}"
if [ -z "${CPU_COUNT}" ]; then
  CPU_COUNT="$(getconf _NPROCESSORS_ONLN)"
fi

GPU_COUNT="${RAY_NUM_GPUS_OVERRIDE:-}"
GPU_MEM_LIST_MB=""

if [ -z "${GPU_COUNT}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_MEM_LIST_MB="$(
      nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
      | tr -d ' ' \
      || true
    )"
    if [ -n "${GPU_MEM_LIST_MB}" ]; then
      GPU_COUNT="$(printf '%s\n' "${GPU_MEM_LIST_MB}" | sed '/^$/d' | wc -l | tr -d ' ')"
    else
      GPU_COUNT="0"
    fi
  else
    GPU_COUNT="0"
  fi
else
  if [ "${GPU_COUNT}" != "0" ] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU_MEM_LIST_MB="$(
      nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
      | tr -d ' ' \
      || true
    )"
  fi
fi

RESOURCES_JSON="$(
  MEM_GB="${MEM_GB}" \
  GPU_COUNT="${GPU_COUNT}" \
  GPU_MEM_LIST_MB="${GPU_MEM_LIST_MB}" \
  EXTRA_RESOURCES_JSON="${RAY_EXTRA_RESOURCES_JSON:-}" \
  GPU_VRAM_THRESHOLDS_GB="${RAY_GPU_VRAM_THRESHOLDS_GB:-16,24,40,48,80}" \
  python - <<'PY'
import json
import os

mem_gb = int(float(os.environ["MEM_GB"]))
gpu_count = int(float(os.environ["GPU_COUNT"]))
gpu_mem_list_raw = os.environ.get("GPU_MEM_LIST_MB", "").strip()
extra_raw = os.environ.get("EXTRA_RESOURCES_JSON", "").strip()
thresholds_raw = os.environ.get("GPU_VRAM_THRESHOLDS_GB", "16,24,40,48,80").strip()

resources = {
    "memory_gb": mem_gb,
}

gpu_mem_mb = []
if gpu_mem_list_raw:
    for line in gpu_mem_list_raw.splitlines():
        line = line.strip()
        if line:
            gpu_mem_mb.append(int(line))

if gpu_count > 0:
    resources["has_gpu"] = 1

if gpu_mem_mb:
    gpu_mem_gb = [mb / 1024.0 for mb in gpu_mem_mb]
    thresholds = [int(x.strip()) for x in thresholds_raw.split(",") if x.strip()]

    for thr in thresholds:
        cnt = sum(1 for g in gpu_mem_gb if g + 1e-9 >= thr)
        if cnt > 0:
            resources[f"gpu_vram_ge_{thr}"] = cnt

    resources["gpu_vram_max_gb"] = int(max(gpu_mem_gb))

if extra_raw:
    extra = json.loads(extra_raw)
    if not isinstance(extra, dict):
        raise ValueError("RAY_EXTRA_RESOURCES_JSON must be a JSON object")
    resources.update(extra)

print(json.dumps(resources, sort_keys=True, separators=(",", ":")))
PY
)"

echo "[ray-worker] head_url=${HEAD_URL} node_ip=${NODE_IP} num_cpus=${CPU_COUNT} num_gpus=${GPU_COUNT} resources=${RESOURCES_JSON}"

exec ray start \
  --address="${HEAD_URL}" \
  --node-ip-address="${NODE_IP}" \
  --num-cpus="${CPU_COUNT}" \
  --num-gpus="${GPU_COUNT}" \
  --resources="${RESOURCES_JSON}" \
  --disable-usage-stats \
  --object-store-memory="${RAY_OBJECT_STORE_MEMORY:-8589934592}" \
  ${RAY_PASSWORD:+--redis-password="${RAY_PASSWORD}"} \
  --block