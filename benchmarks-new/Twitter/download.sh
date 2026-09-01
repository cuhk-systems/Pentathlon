#!/bin/bash

# Download and decompress a Twitter twemcache cluster trace.
#
# Usage:
#   ./download.sh <clusterN>
#   Example: ./download.sh cluster3

set -euo pipefail

if [[ $# -ne 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <clusterN>"
    echo "  Download and decompress the Twitter twemcache cluster<N> trace."
    echo "  Example: $0 cluster3"
    exit 1
fi

cluster="$1"
base_url="https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source/cluster1.sort.zst"
url="${base_url/cluster1/$cluster}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
file="$cluster.sort.zst"

echo "Downloading $url ..."
wget -O "$script_dir/$file" "$url"

echo "Decompressing $file ..."
zstd -d "$script_dir/$file"
