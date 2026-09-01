#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CLANG="${CLANG:-clang}"
CLANGXX="${CLANGXX:-clang++}"
LLVM_OPT="${LLVM_OPT:-$REPO_ROOT/compiler_new/build/bin/my-llvm-opt}"
LLVM_AR="${LLVM_AR:-llvm-ar}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
USE_CXL="${PENTATHLON_USE_CXL:-ON}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-}"

SKIPLIST_DIR="$REPO_ROOT/benchmarks-new/skiplist"
BUILD_DIR="$SKIPLIST_DIR/build"
INPUT_CPP="$SKIPLIST_DIR/main.cpp"
INPUT_LL="$BUILD_DIR/skiplist.input.ll"
OPT_LL="$SKIPLIST_DIR/skiplist.optimized.ll"
OBJ_OUT="$BUILD_DIR/skiplist.o"
LIB_OUT="$BUILD_DIR/libskiplist.a"
YCSB_DIR="$REPO_ROOT/benchmarks-new/YCSB"
BENCHMARK_NAME="benchmark-skiplist"
BENCHMARK_PROJECT_DIR="$BUILD_DIR/compute_new_ycsb_project"
BENCHMARK_BUILD_DIR="$BUILD_DIR/compute_new_ycsb_build"
BENCHMARK_CMAKE="$BENCHMARK_PROJECT_DIR/CMakeLists.txt"
BENCHMARK_OUT_DIR="$BENCHMARK_BUILD_DIR/bin"
BENCHMARK_OUT="$BENCHMARK_OUT_DIR/$BENCHMARK_NAME"

OPT_FLAGS=(
  --addr-dep-pass
  --disagg-alloc-pass
  --disagg-free-pass
  --mark-dirty-pass
  --addr-dep-rel-pass
  --local-addr-pass
)

mkdir -p "$BUILD_DIR"

if [[ ! -f "$INPUT_CPP" ]]; then
  echo "Missing source file: $INPUT_CPP" >&2
  exit 1
fi

if ! command -v "$CLANG" >/dev/null 2>&1; then
  echo "Missing clang: $CLANG" >&2
  exit 1
fi

if ! command -v "$CLANGXX" >/dev/null 2>&1; then
  echo "Missing clang++: $CLANGXX" >&2
  exit 1
fi

if ! command -v "$LLVM_AR" >/dev/null 2>&1; then
  echo "Missing llvm-ar: $LLVM_AR" >&2
  exit 1
fi

if ! command -v "$CMAKE_BIN" >/dev/null 2>&1; then
  echo "Missing cmake: $CMAKE_BIN" >&2
  exit 1
fi

if [[ -z "$CMAKE_GENERATOR" ]]; then
  if command -v ninja >/dev/null 2>&1; then
    CMAKE_GENERATOR="Ninja"
  else
    CMAKE_GENERATOR="Unix Makefiles"
  fi
fi

echo "Compiling skiplist benchmark source to LLVM IR..."
"$CLANGXX" -std=c++17 -O0 -S -emit-llvm "$INPUT_CPP" -o "$INPUT_LL" -pthread

if [[ -x "$LLVM_OPT" ]]; then
  echo "Running compiler_new passes..."
  "$LLVM_OPT" "${OPT_FLAGS[@]}" "$INPUT_LL" -o "$OPT_LL"
else
  echo "Warning: missing optimizer $LLVM_OPT; using unoptimized LLVM IR" >&2
  cp "$INPUT_LL" "$OPT_LL"
fi

echo "Compiling LLVM IR to object..."
"$CLANGXX" -O3 -c -g "$OPT_LL" -o "$OBJ_OUT"

echo "Creating static library..."
rm -f "$LIB_OUT"
"$LLVM_AR" rcs "$LIB_OUT" "$OBJ_OUT"

echo "Generating standalone compute_new benchmark project..."
mkdir -p "$BENCHMARK_PROJECT_DIR" "$BENCHMARK_OUT_DIR"
cat >"$BENCHMARK_CMAKE" <<EOF
cmake_minimum_required(VERSION 3.20)
project(pentathlon_skiplist_compute_new_ycsb LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "$BENCHMARK_OUT_DIR")
set(PENTATHLON_USE_COMPUTE_NEW ON CACHE BOOL "" FORCE)
set(PENTATHLON_USE_CXL "$USE_CXL" CACHE BOOL "" FORCE)

find_package(Threads REQUIRED)

add_subdirectory("$REPO_ROOT/runtime" runtime)

add_executable($BENCHMARK_NAME
  "$YCSB_DIR/ycsb.c"
  "$YCSB_DIR/workload.c"
  "$YCSB_DIR/generators/splitmix64.c"
  "$YCSB_DIR/generators/xoshiro256p.c"
  "$YCSB_DIR/generators/zipf.c"
)

set_target_properties($BENCHMARK_NAME PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories($BENCHMARK_NAME
  PRIVATE
    "$YCSB_DIR"
)

target_link_libraries($BENCHMARK_NAME
  PRIVATE
    "$LIB_OUT"
    dm_compiler_rt_compute
    Threads::Threads
)
EOF

echo "Configuring standalone compute_new benchmark build..."
"$CMAKE_BIN" -S "$BENCHMARK_PROJECT_DIR" -B "$BENCHMARK_BUILD_DIR" -G "$CMAKE_GENERATOR" \
  -DCMAKE_MODULE_PATH="$REPO_ROOT/scripts/cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CLANG" \
  -DCMAKE_CXX_COMPILER="$CLANGXX"

echo "Building $BENCHMARK_NAME..."
"$CMAKE_BIN" --build "$BENCHMARK_BUILD_DIR" --target "$BENCHMARK_NAME"

echo "Built library: $LIB_OUT"
echo "Built benchmark: $BENCHMARK_OUT"
