#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -gt 0 ]]; then
    targets=("$@")
else
    targets=(bptree hashtable skiplist bptree-twitter hashtable-twitter skiplist-twitter)
fi

for target in "${targets[@]}"; do
    case "$target" in
        bptree|hashtable|skiplist)
            build_script="$SCRIPT_DIR/build_${target}_compute_new_ycsb.sh"
            label="$target YCSB"
            ;;
        bptree-ycsb|hashtable-ycsb|skiplist-ycsb)
            benchmark="${target%-ycsb}"
            build_script="$SCRIPT_DIR/build_${benchmark}_compute_new_ycsb.sh"
            label="$benchmark YCSB"
            ;;
        bptree-twitter|hashtable-twitter|skiplist-twitter|twitter-bptree|twitter-hashtable|twitter-skiplist)
            benchmark="${target%-twitter}"
            benchmark="${benchmark#twitter-}"
            build_script="$SCRIPT_DIR/build_${benchmark}_compute_new_twitter_trace.sh"
            label="$benchmark Twitter"
            ;;
        *)
            echo "Unknown build target: $target" >&2
            exit 2
            ;;
    esac

    echo "Building $label (RDMA)..."
    PENTATHLON_USE_CXL=OFF bash "$build_script"
done
