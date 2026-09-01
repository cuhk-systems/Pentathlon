# Pentathlon

## Contents

- `compiler_new/`
- `runtime/compute_new/`
- `runtime/memory/`
- `runtime/common/`
- `benchmarks-new/`
- `scripts/`

## Requirements

- Linux on x86-64
- CMake 3.20 or newer
- Clang/LLVM 18 and `llvm-ar`
- a C++17 compiler
- Ninja or Make
- `libnuma`, POSIX threads, `bc`, and GNU `timeout`
- `wget` and `zstd` for the Twitter workload
- `libibverbs`, `librdmacm` for RDMA builds

Set `CLANG`, `CLANGXX`, `LLVM_AR`, and `CMAKE_BIN` if these tools are not on
`PATH`.

## Setup

Make the supplied scripts executable:

```bash
chmod +x scripts/*.sh
```

Fetch a Twitter trace when needed:

```bash
bash benchmarks-new/Twitter/download.sh cluster3
```

## Build

Build the compiler tools:

```bash
cmake -S compiler_new -B compiler_new/build -G Ninja \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
cmake --build compiler_new/build
```

Build all supplied benchmarks and workloads for CXL:

```bash
bash scripts/build_cxl.sh
```

Build all supplied benchmarks and workloads for RDMA:

```bash
bash scripts/build_rdma.sh
```

Pass benchmark names to build only selected targets:

```bash
bash scripts/build_cxl.sh bptree hashtable
bash scripts/build_rdma.sh skiplist
bash scripts/build_cxl.sh bptree-twitter hashtable-twitter
bash scripts/build_rdma.sh skiplist-twitter
```

## Run

Run YCSB:

```bash
bash scripts/run_ycsb.sh bptree results/bptree-ycsb.csv
```

Run Twitter:

```bash
bash scripts/run_twitter.sh bptree results/bptree-twitter.csv
```

Select thread counts or set a timeout:

```bash
PTH_BM_THREADS="1 2 4 8 16" PTH_BM_TIMEOUT=1800 \
  bash scripts/run_ycsb.sh hashtable results/hashtable-ycsb.csv
```
