#!/usr/bin/env bash
# 在 tmux 里挂着跑：按 worker / shard 打 Redis pose 队列占用。
#   ./scripts/monitor-pose-lag.sh            # 每 300s，一直跑
#   ./scripts/monitor-pose-lag.sh 300 24h    # 跑 24 小时后退出
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

INTERVAL="${1:-${INTERVAL:-300}}"
DURATION_RAW="${2:-${DURATION:-forever}}"
LOGICAL_SHARDS="${POSE_LOGICAL_SHARD_COUNT:-16}"
REDIS_CONTAINER="${REDIS_CONTAINER:-visual-dps-redis}"
GROUP="${POSE_STREAM_GROUP:-event-workers}"
PREFIX="${POSE_STREAM_KEY_PREFIX:-pose:stream}"
WORKER_A="${WORKER_A:-visual-dps-event-worker}"
WORKER_B="${WORKER_B:-visual-dps-event-worker-b}"
A_START="${EVENT_WORKER_SHARD_START:-0}"
A_END="${EVENT_WORKER_SHARD_END:-7}"
B_START="${EVENT_WORKER_B_SHARD_START:-8}"
B_END="${EVENT_WORKER_B_SHARD_END:-15}"
WARN_LAG="${LAG_MONITOR_WARN:-50}"
CRIT_LAG="${LAG_MONITOR_CRIT:-500}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"

parse_duration() {
  local s="$1"
  case "$s" in
    forever|0|"") echo 0 ;;
    *d) echo $((${s%d} * 86400)) ;;
    *h) echo $((${s%h} * 3600)) ;;
    *m) echo $((${s%m} * 60)) ;;
    *s) echo "${s%s}" ;;
    *) echo "$s" ;;
  esac
}

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif sudo docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
else
  echo "docker 不可用" >&2
  exit 1
fi

DURATION_S="$(parse_duration "$DURATION_RAW")"
mkdir -p "$LOG_DIR"
START="$(date '+%Y-%m-%d %H:%M:%S')"
LOG_FILE="${LOG_DIR}/pose-lag-$(date +%Y%m%d).log"
if [[ "$DURATION_S" -eq 0 ]]; then
  DUR_TXT="forever"
else
  DUR_TXT="$DURATION_RAW"
fi

redis_cmd() {
  if [[ -n "${REDIS_PASSWORD:-}" ]]; then
    "${DOCKER[@]}" exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" --no-auth-warning "$@"
  else
    "${DOCKER[@]}" exec "$REDIS_CONTAINER" redis-cli "$@"
  fi
}

# entries-read 空行时不能把下一字段 lag 当成值。
parse_xinfo() {
  awk '
    /^pending$/      { getline v; if (v ~ /^[0-9]+$/) pending=v }
    /^lag$/          { getline v; if (v ~ /^[0-9]+$/) lag=v }
    /^entries-read$/ { getline v; if (v ~ /^[0-9]+$/) read=v }
    END { printf "%d %d %d\n", lag+0, pending+0, read+0 }
  '
}

declare -a CUR_LAG CUR_PENDING CUR_XLEN CUR_READ
declare -a PREV_LAG PREV_PENDING PREV_XLEN PREV_READ

snapshot() {
  local sid raw xl sl sp sr
  for ((sid = 0; sid < LOGICAL_SHARDS; sid++)); do
    raw="$(redis_cmd XINFO GROUPS "${PREFIX}:${sid}" 2>/dev/null || true)"
    read -r sl sp sr <<<"$(printf '%s\n' "$raw" | parse_xinfo)"
    xl="$(redis_cmd XLEN "${PREFIX}:${sid}" 2>/dev/null | tail -n1 || true)"
    [[ "$xl" =~ ^[0-9]+$ ]] || xl=0
    CUR_LAG[sid]="$sl"
    CUR_PENDING[sid]="$sp"
    CUR_XLEN[sid]="$xl"
    CUR_READ[sid]="$sr"
  done
}

infer_count() {
  "${DOCKER[@]}" ps --format '{{.Names}}' --filter 'name=visual-dps-infer' 2>/dev/null \
    | grep -c '^visual-dps-infer' || true
}

# 输出: cpu_a cpu_b
worker_cpus() {
  "${DOCKER[@]}" stats --no-stream --format '{{.Name}} {{.CPUPerc}}' "$WORKER_A" "$WORKER_B" 2>/dev/null \
    | awk -v a="$WORKER_A" -v b="$WORKER_B" '
        $1 == a { ga = $2 }
        $1 == b { gb = $2 }
        END {
          if (ga == "") ga = "-"
          if (gb == "") gb = "-"
          print ga, gb
        }'
}

flag_of() {
  local lag="$1" d_lag="$2" baseline="$3"
  if [[ "$baseline" == 1 ]]; then
    echo baseline
    return
  fi
  if (( lag >= CRIT_LAG )); then
    echo CRITICAL
  elif (( lag >= WARN_LAG || d_lag >= WARN_LAG )); then
    echo WARN
  fi
}

emit() {
  local line="$1"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >>"$LOG_FILE"
}

# worker 汇总一行 + 其下每个 shard 一行
emit_worker() {
  local now="$1" name="$2" start="$3" end="$4" cpu="$5" baseline="$6" dt="$7"
  local sid lag=0 pending=0 xlen=0 read=0 prev_lag=0 prev_read=0
  local dr dl consume ingress flag shard_lbl
  local s_lag s_pending s_xlen s_read s_prev_lag s_prev_read s_dr s_dl s_consume s_ingress s_flag

  for ((sid = start; sid <= end; sid++)); do
    lag=$((lag + CUR_LAG[sid]))
    pending=$((pending + CUR_PENDING[sid]))
    xlen=$((xlen + CUR_XLEN[sid]))
    read=$((read + CUR_READ[sid]))
    prev_lag=$((prev_lag + PREV_LAG[sid]))
    prev_read=$((prev_read + PREV_READ[sid]))
  done
  dr=$((read - prev_read))
  dl=$((lag - prev_lag))
  (( dr < 0 )) && dr=0
  if [[ "$baseline" == 1 || "$dt" -le 0 ]]; then
    consume="-"
    ingress="-"
    dl_txt="-"
    flag="baseline"
  else
    consume="$(awk -v d="$dr" -v t="$dt" 'BEGIN { printf "%.2f", d / t }')"
    ingress="$(awk -v d="$dr" -v l="$dl" -v t="$dt" 'BEGIN { printf "%.2f", (d + l) / t }')"
    dl_txt="$dl"
    flag="$(flag_of "$lag" "$dl" 0)"
  fi
  shard_lbl="${start}-${end}"
  printf -v LINE '%s %-6s %5s %8s %8s %8d %6s %7d %5d %5s %10s %s' \
    "$now" "$name" "$shard_lbl" "$consume" "$ingress" "$lag" "$dl_txt" "$pending" "$xlen" "-" "$cpu" "$flag"
  emit "$LINE"

  for ((sid = start; sid <= end; sid++)); do
    s_lag="${CUR_LAG[sid]}"
    s_pending="${CUR_PENDING[sid]}"
    s_xlen="${CUR_XLEN[sid]}"
    s_read="${CUR_READ[sid]}"
    s_prev_lag="${PREV_LAG[sid]}"
    s_prev_read="${PREV_READ[sid]}"
    s_dr=$((s_read - s_prev_read))
    s_dl=$((s_lag - s_prev_lag))
    (( s_dr < 0 )) && s_dr=0
    if [[ "$baseline" == 1 || "$dt" -le 0 ]]; then
      s_consume="-"
      s_ingress="-"
      s_dl_txt="-"
      s_flag=""
    else
      s_consume="$(awk -v d="$s_dr" -v t="$dt" 'BEGIN { printf "%.2f", d / t }')"
      s_ingress="$(awk -v d="$s_dr" -v l="$s_dl" -v t="$dt" 'BEGIN { printf "%.2f", (d + l) / t }')"
      s_dl_txt="$s_dl"
      s_flag="$(flag_of "$s_lag" "$s_dl" 0)"
    fi
    printf -v LINE '%s %-6s %5s %8s %8s %8d %6s %7d %5d %5s %10s %s' \
      "$now" "$name" "$sid" "$s_consume" "$s_ingress" "$s_lag" "$s_dl_txt" "$s_pending" "$s_xlen" "-" "-" "$s_flag"
    emit "$LINE"
  done
}

copy_prev() {
  local sid
  for ((sid = 0; sid < LOGICAL_SHARDS; sid++)); do
    PREV_LAG[sid]="${CUR_LAG[sid]:-0}"
    PREV_PENDING[sid]="${CUR_PENDING[sid]:-0}"
    PREV_XLEN[sid]="${CUR_XLEN[sid]:-0}"
    PREV_READ[sid]="${CUR_READ[sid]:-0}"
  done
}

{
  echo "============================================================"
  echo "Visual-DPS lag monitor"
  echo "  deploy: $ROOT"
  echo "  app:    $ROOT"
  echo "  log:    $LOG_FILE"
  echo "  start:  $START"
  echo "  every:  ${INTERVAL}s   duration: $DUR_TXT"
  echo "  redis:  $REDIS_CONTAINER   group: $GROUP"
  echo "  worker A: $WORKER_A   shards ${A_START}-${A_END}"
  echo "  worker B: $WORKER_B   shards ${B_START}-${B_END}"
  echo "  stream: ${PREFIX}:0-$((LOGICAL_SHARDS - 1))"
  echo "  consume=worker msg/s  ingress=produce msg/s  d_lag=lag delta this period"
  echo "============================================================"
  echo "time                 worker shard consume  ingress      lag  d_lag pending  xlen infer        cpu flag"
  echo "------------------- ------ ----- -------- -------- -------- ------ ------- ----- ----- ---------- ----"
} | tee -a "$LOG_FILE"

init_prev() {
  local sid
  for ((sid = 0; sid < LOGICAL_SHARDS; sid++)); do
    PREV_LAG[sid]=0
    PREV_PENDING[sid]=0
    PREV_XLEN[sid]=0
    PREV_READ[sid]=0
  done
}

emit_tick() {
  local baseline="$1" dt="$2"
  local now infer cpu_a cpu_b
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  infer="$(infer_count)"
  read -r cpu_a cpu_b <<<"$(worker_cpus)"
  emit "---- ${now}  infer=${infer} ----"
  emit_worker "$now" "A" "$A_START" "$A_END" "$cpu_a" "$baseline" "$dt"
  emit_worker "$now" "B" "$B_START" "$B_END" "$cpu_b" "$baseline" "$dt"
}

init_prev
PREV_TS="$(date +%s)"
snapshot
emit_tick 1 0
copy_prev
PREV_TS="$(date +%s)"

DEADLINE=0
if [[ "$DURATION_S" -gt 0 ]]; then
  DEADLINE=$((PREV_TS + DURATION_S))
fi

while true; do
  if [[ "$DEADLINE" -gt 0 && "$(date +%s)" -ge "$DEADLINE" ]]; then
    break
  fi
  sleep "$INTERVAL"
  now_s="$(date +%s)"
  snapshot
  dt=$((now_s - PREV_TS))
  if (( dt <= 0 )); then
    continue
  fi
  emit_tick 0 "$dt"
  copy_prev
  PREV_TS="$now_s"
done
