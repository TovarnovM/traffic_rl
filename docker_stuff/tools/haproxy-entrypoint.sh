#!/bin/sh
set -e

: "${OPTUNA_PROXY_PORT_MIN:?Need OPTUNA_PROXY_PORT_MIN}"
: "${OPTUNA_PROXY_PORT_MAX:?Need OPTUNA_PROXY_PORT_MAX}"
: "${OPTUNA_PROXY_HOST:=optuna-proxies}"

THREAD_POOL_SIZE=${OPTUNA_PROXY_THREAD_POOL_SIZE:-32}

CFG=/tmp/haproxy.cfg   # <-- важно: /tmp writable

cat > "$CFG" <<EOF
global
  maxconn $((THREAD_POOL_SIZE * 512))
  log stdout format raw local0

defaults
  mode tcp
  option tcplog
  timeout connect 5s
  timeout client  30m
  timeout server  30m

frontend optuna_grpc_front
  bind *:13000
  default_backend optuna_grpc_back

backend optuna_grpc_back
  balance roundrobin
EOF

port="$OPTUNA_PROXY_PORT_MIN"
while [ "$port" -le "$OPTUNA_PROXY_PORT_MAX" ]; do
  echo "  server proxy_${port} ${OPTUNA_PROXY_HOST}:${port} check" >> "$CFG"
  port=$((port + 1))
done

echo "Generated haproxy.cfg:"
cat "$CFG"

exec haproxy -f "$CFG"   # <-- запускаем с /tmp
