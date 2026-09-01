#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  echo "Usage: $0 <bptree|hashtable|skiplist> [output.csv]" >&2
  echo "Environment: PTH_BM_THREADS, PTH_BM_UNIFORMS, PTH_BM_WORKLOADS, PTH_BM_TIMEOUT" >&2
}

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  usage
  exit 2
fi

benchmark="$1"
output_file="${2:-$REPO_ROOT/results/${benchmark}-ycsb.csv}"

case "$benchmark" in
  bptree|hashtable|skiplist) ;;
  *)
    usage
    exit 2
    ;;
esac

binary="$REPO_ROOT/benchmarks-new/$benchmark/build/compute_new_ycsb_build/bin/benchmark-$benchmark"
if [[ ! -x "$binary" ]]; then
  echo "Missing benchmark binary: $binary" >&2
  echo "Build it first with: bash scripts/build_cxl.sh $benchmark" >&2
  exit 1
fi

threads="${PTH_BM_THREADS:-1 2 4 8 16}"
uniforms="${PTH_BM_UNIFORMS:-0 1}"
workloads="${PTH_BM_WORKLOADS:-read-only read-intensive write-intensive}"
timeout_secs="${PTH_BM_TIMEOUT:-}"

mkdir -p "$(dirname "$output_file")"

run_one() {
  local uniform="$1"
  local thread_num="$2"
  local workload="$3"
  local read_ratio
  local insert_ratio

  case "$workload" in
    read-only)
      read_ratio=100
      insert_ratio=0
      ;;
    read-intensive)
      read_ratio=95
      insert_ratio=5
      ;;
    write-intensive)
      read_ratio=50
      insert_ratio=50
      ;;
    *)
      echo "Unknown YCSB workload: $workload" >&2
      exit 2
      ;;
  esac

  echo "Running $benchmark YCSB workload=$workload uniform=$uniform threads=$thread_num"
  if [[ -n "$timeout_secs" ]]; then
    timeout "$timeout_secs" env \
      PTH_BM_FILENAME="$output_file" \
      PTH_BM_EXTRA_COLS="benchmark,workload" \
      PTH_BM_EXTRA_COL_VALUES="$benchmark,$workload" \
      PTH_BM_UNIFORM="$uniform" \
      PTH_BM_THREAD_NUM="$thread_num" \
      PTH_BM_READ_RATIO="$read_ratio" \
      PTH_BM_INSERT_RATIO="$insert_ratio" \
      "$binary"
  else
    env \
      PTH_BM_FILENAME="$output_file" \
      PTH_BM_EXTRA_COLS="benchmark,workload" \
      PTH_BM_EXTRA_COL_VALUES="$benchmark,$workload" \
      PTH_BM_UNIFORM="$uniform" \
      PTH_BM_THREAD_NUM="$thread_num" \
      PTH_BM_READ_RATIO="$read_ratio" \
      PTH_BM_INSERT_RATIO="$insert_ratio" \
      "$binary"
  fi
}

for uniform in $uniforms; do
  for workload in $workloads; do
    for thread_num in $threads; do
      run_one "$uniform" "$thread_num" "$workload"
    done
  done
done

echo "Results written to: $output_file"
