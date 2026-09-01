#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=""
BUILD_DIR=""
BUILD_TYPE="RelWithDebInfo"
MEMORY_ADDR_ARG=""
JOBS=8
RUN_BATCH_READ=0
BUILD_TARGET="compute_new-test-binaries"
MEMORY_TARGET="dm_compiler_rt_memory"
MEMORY_PORT="12345"
MEMORY_POOL_SIZE=""
START_MEMORY_SIDE=1
MEMORY_LOG=""
USE_CXL="${PENTATHLON_USE_CXL:-1}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-}"

MEMORY_PID=""

usage() {
  cat <<'EOF'
Usage: run_compute_new_tests.sh [options]

Options:
  --repo-root <path>      Repository root path (default: auto-detect)
  --build-dir <path>      Build directory (default: <repo>/runtime/build_compute_new)
  --build-type <type>     CMake build type (default: RelWithDebInfo)
  --generator <name>      CMake generator (default: Ninja if available, else Unix Makefiles)
  --memory-addr <addr>    MEMORY_ADDR for runtime init
  --jobs <N>              Parallel build jobs (default: 8)
  --build-target <name>   CMake target to build (default: compute_new-test-binaries)
  --memory-target <name>  Memory-side target to build (default: dm_compiler_rt_memory)
  --memory-port <port>    Memory-side port (default: 12345)
  --memory-pool-size <s>  PTH_MEMORY_POOL_SIZE for memory side, e.g. 4G
  --memory-log <path>     Memory-side log path (default: <build-dir>/dm_compiler_rt_memory.log)
  --no-start-memory       Do not start local memory side; assume it is already running
  --cxl                   Build/run CXL NUMA backend (default on this branch)
  --rdma                  Build/run RDMA backend
  --run-batch-read        Also run test-batch-read
  -h, --help              Show this help
EOF
}

cleanup_memory() {
  if [[ -n "${MEMORY_PID:-}" ]]; then
    if kill -0 "$MEMORY_PID" 2>/dev/null; then
      kill "$MEMORY_PID" 2>/dev/null || true
      wait "$MEMORY_PID" 2>/dev/null || true
    fi
    MEMORY_PID=""
  fi
}

print_memory_log_on_failure() {
  if [[ -n "${MEMORY_LOG:-}" && -f "$MEMORY_LOG" ]]; then
    echo
    echo "== Memory log tail =="
    tail -n 50 "$MEMORY_LOG" || true
  fi
}

start_memory_side() {
  local memory_bin="$BUILD_DIR/memory/$MEMORY_TARGET"
  if [[ ! -x "$memory_bin" ]]; then
    echo "Memory binary not found or not executable: $memory_bin" >&2
    exit 1
  fi

  : > "$MEMORY_LOG"

  local -a memory_cmd=("$memory_bin" "--port" "$MEMORY_PORT")
  if [[ -n "$MEMORY_POOL_SIZE" ]]; then
    PTH_MEMORY_POOL_SIZE="$MEMORY_POOL_SIZE" "${memory_cmd[@]}" >"$MEMORY_LOG" 2>&1 &
  else
    "${memory_cmd[@]}" >"$MEMORY_LOG" 2>&1 &
  fi
  MEMORY_PID=$!

  for _ in {1..50}; do
    if ! kill -0 "$MEMORY_PID" 2>/dev/null; then
      echo "Memory side exited during startup." >&2
      print_memory_log_on_failure
      exit 1
    fi
    if grep -q "Waiting for connection from compute..." "$MEMORY_LOG" 2>/dev/null; then
      return 0
    fi
    sleep 0.2
  done

  echo "Timed out waiting for memory side to initialize." >&2
  print_memory_log_on_failure
  exit 1
}

trap cleanup_memory EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    --generator)
      CMAKE_GENERATOR="$2"
      shift 2
      ;;
    --memory-addr)
      MEMORY_ADDR_ARG="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --build-target)
      BUILD_TARGET="$2"
      shift 2
      ;;
    --memory-target)
      MEMORY_TARGET="$2"
      shift 2
      ;;
    --memory-port)
      MEMORY_PORT="$2"
      shift 2
      ;;
    --memory-pool-size)
      MEMORY_POOL_SIZE="$2"
      shift 2
      ;;
    --memory-log)
      MEMORY_LOG="$2"
      shift 2
      ;;
    --no-start-memory)
      START_MEMORY_SIDE=0
      shift
      ;;
    --cxl)
      USE_CXL=1
      START_MEMORY_SIDE=0
      shift
      ;;
    --rdma)
      USE_CXL=0
      shift
      ;;
    --run-batch-read)
      RUN_BATCH_READ=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "${USE_CXL,,}" in
  1|on|true|yes)
    USE_CXL=1
    ;;
  0|off|false|no)
    USE_CXL=0
    ;;
  *)
    echo "PENTATHLON_USE_CXL must be one of 1/0, on/off, true/false, yes/no" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
else
  REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
fi

RUNTIME_DIR="$REPO_ROOT/runtime"
if [[ -z "$CMAKE_GENERATOR" ]]; then
  if command -v ninja >/dev/null 2>&1; then
    CMAKE_GENERATOR="Ninja"
  else
    CMAKE_GENERATOR="Unix Makefiles"
  fi
fi

if [[ -z "$BUILD_DIR" ]]; then
  if [[ "$USE_CXL" -eq 1 ]]; then
    if [[ "$CMAKE_GENERATOR" == "Ninja" ]]; then
      BUILD_DIR="$RUNTIME_DIR/build_compute_new_cxl"
    else
      BUILD_DIR="$RUNTIME_DIR/build_compute_new_cxl_make"
    fi
  else
    BUILD_DIR="$RUNTIME_DIR/build_compute_new"
  fi
fi

if [[ -z "$MEMORY_LOG" ]]; then
  MEMORY_LOG="$BUILD_DIR/dm_compiler_rt_memory.log"
fi

if [[ "$USE_CXL" -eq 1 ]]; then
  START_MEMORY_SIDE=0
fi

if [[ -n "$MEMORY_ADDR_ARG" ]]; then
  export MEMORY_ADDR="$MEMORY_ADDR_ARG"
fi
if [[ "$USE_CXL" -eq 0 && "$START_MEMORY_SIDE" -eq 1 && -z "${MEMORY_ADDR:-}" ]]; then
  export MEMORY_ADDR="127.0.0.1"
fi
if [[ "$USE_CXL" -eq 0 && -z "${MEMORY_ADDR:-}" ]]; then
  echo "MEMORY_ADDR is not set. Pass --memory-addr <ip-or-host>, export MEMORY_ADDR first, or allow local memory startup." >&2
  exit 1
fi

echo "Repo root: $REPO_ROOT"
echo "Runtime dir: $RUNTIME_DIR"
echo "Build dir: $BUILD_DIR"
echo "Build type: $BUILD_TYPE"
echo "CMake generator: $CMAKE_GENERATOR"
echo "Build target: $BUILD_TARGET"
echo "CXL backend: $USE_CXL"
echo "Memory target: $MEMORY_TARGET"
echo "Memory port: $MEMORY_PORT"
echo "Start memory side: $START_MEMORY_SIDE"
if [[ "$USE_CXL" -eq 0 ]]; then
  echo "MEMORY_ADDR: $MEMORY_ADDR"
fi
echo "Memory log: $MEMORY_LOG"
if [[ -n "$MEMORY_POOL_SIZE" ]]; then
  echo "PTH_MEMORY_POOL_SIZE: $MEMORY_POOL_SIZE"
fi

cmake_args=(
  -S "$RUNTIME_DIR"
  -B "$BUILD_DIR"
  -G "$CMAKE_GENERATOR"
  -DPENTATHLON_USE_COMPUTE_NEW=ON
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)
if [[ "$USE_CXL" -eq 1 ]]; then
  cmake_args+=(-DPENTATHLON_USE_CXL=ON)
fi
cmake "${cmake_args[@]}"

cmake --build "$BUILD_DIR" --target "$BUILD_TARGET" -j "$JOBS"
if [[ "$START_MEMORY_SIDE" -eq 1 ]]; then
  cmake --build "$BUILD_DIR" --target "$MEMORY_TARGET" -j "$JOBS"
fi

tests=(
  "dm_compiler_rt_compute_tests"
  "dm_compiler_rt_compute_tests_c"
)
if [[ "$RUN_BATCH_READ" -eq 1 ]]; then
  tests+=("test-batch-read")
fi

for test_name in "${tests[@]}"; do  if [[ "$START_MEMORY_SIDE" -eq 1 ]]; then
    cleanup_memory
    start_memory_side
  fi

  echo
  echo "== Running $test_name =="
  test_path="$BUILD_DIR/compute_new/$test_name"
  if [[ ! -x "$test_path" ]]; then
    echo "Test binary not found or not executable: $test_path" >&2
    exit 1
  fi

  if ! "$test_path"; then
    echo "$test_name failed." >&2
    if [[ "$START_MEMORY_SIDE" -eq 1 ]]; then
      print_memory_log_on_failure
    fi
    exit 1
  fi

  if [[ "$START_MEMORY_SIDE" -eq 1 ]]; then
    cleanup_memory
  fi
done

echo
echo "All selected compute_new tests passed."
