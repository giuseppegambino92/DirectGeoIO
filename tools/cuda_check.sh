#!/usr/bin/env bash
# Compile-check the CUDA backend (nvcc kernels + cuFile wrapper + Rust bindings) inside an
# NVIDIA CUDA devel container. No GPU is needed to compile; running requires an NVIDIA GPU.
#   tools/cuda_check.sh [image]
set -euo pipefail
IMAGE=${1:-nvidia/cuda:13.0.2-devel-ubuntu24.04}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
docker run --rm -v "$ROOT":/src:ro -e CARGO_TARGET_DIR=/tmp/target "$IMAGE" bash -lc '
  set -euo pipefail
  apt-get update -qq && apt-get install -y -qq curl ca-certificates build-essential >/dev/null
  curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal >/dev/null
  source $HOME/.cargo/env
  nvcc --version | tail -2
  echo "cufile.h: $(ls /usr/local/cuda/include/cufile.h 2>/dev/null || echo absent)"
  echo "== nvcc: kernels + host runtime (all target archs) =="
  cd /src && nvcc -O3 -std=c++17 --fmad=false -Xcompiler -Wall -c kernels/cuda/host.cu -o /tmp/host.o \
     -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_120,code=sm_120
  cuobjdump --list-elf /tmp/host.o | head
  echo "== cargo build --release -p directgeoio --features cuda =="
  cargo build --release -p directgeoio --features cuda 2>&1 | tail -5
  ls -la /tmp/target/release/libdirectgeoio.so
  nm -D /tmp/target/release/libdirectgeoio.so | grep -c " T dgio_" | xargs echo "exported dgio_ symbols:"
  echo CUDA_CHECK_OK
'
