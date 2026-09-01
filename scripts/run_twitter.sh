#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  echo "Usage: $0 <bptree|hashtable|skiplist> [output.csv]" >&2
  echo "Environment: PTH_BM_THREADS, PTH_BM_WORKLOAD, PTH_BM_TIMEOUT" >&2
}

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  usage
  exit 2
fi

benchmark="$1"
output_file="${2:-$REPO_ROOT/results/${benchmark}-twitter.csv}"

case "$benchmark" in
  bptree|hashtable|skiplist) ;;
  *)
    usage
    exit 2
    ;;
esac

binary="$REPO_ROOT/benchmarks-new/$benchmark/build/compute_new_twitter_trace_build/bin/benchmark-$benchmark-twitter-trace"
workload="${PTH_BM_WORKLOAD:-cluster3.sort}"
workload_path="$REPO_ROOT/benchmarks-new/Twitter/$workload"
threads="${PTH_BM_THREADS:-1 2 4 8 16}"
timeout_secs="${PTH_BM_TIMEOUT:-}"

if [[ ! -x "$binary" ]]; then
  echo "Missing benchmark binary: $binary" >&2
  echo "Build it first with: bash scripts/build_cxl.sh ${benchmark}-twitter" >&2
  exit 1
fi

if [[ ! -f "$workload_path" ]]; then
  echo "Missing Twitter workload: $workload_path" >&2
  echo "Fetch it with: bash benchmarks-new/Twitter/download.sh cluster3" >&2
  exit 1
fi

mkdir -p "$(dirname "$output_file")"

for thread_num in $threads; do
  echo "Running $benchmark Twitter workload=$workload threads=$thread_num"
  if [[ -n "$timeout_secs" ]]; then
    timeout "$timeout_secs" env \
      PTH_BM_FILENAME="$output_file" \
      PTH_BM_EXTRA_COLS="benchmark" \
      PTH_BM_EXTRA_COL_VALUES="$benchmark" \
      PTH_BM_WORKLOAD="$workload" \
      PTH_BM_THREAD_NUM="$thread_num" \
      "$binary"
  else
    env \
      PTH_BM_FILENAME="$output_file" \
      PTH_BM_EXTRA_COLS="benchmark" \
      PTH_BM_EXTRA_COL_VALUES="$benchmark" \
      PTH_BM_WORKLOAD="$workload" \
      PTH_BM_THREAD_NUM="$thread_num" \
      "$binary"
  fi
done

echo "Results written to: $output_file"
