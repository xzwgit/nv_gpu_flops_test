#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_FILE="${ROOT_DIR}/src/gpu_dense_bench.cu"
BUILD_DIR="${ROOT_DIR}/build"
BIN_FILE="${BUILD_DIR}/gpu_dense_bench"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Install a working NVIDIA driver."
command -v nvcc >/dev/null 2>&1 || die "nvcc not found. Install the CUDA Toolkit and add its bin directory to PATH."

mkdir -p "${BUILD_DIR}"

arch_flags=()
while IFS= read -r cap; do
  cap="${cap//[[:space:]]/}"
  cap="${cap//./}"
  if [[ "${cap}" =~ ^[0-9]+$ ]]; then
    flag="-gencode=arch=compute_${cap},code=sm_${cap}"
    duplicate=0
    for existing in "${arch_flags[@]}"; do
      if [[ "${existing}" == "${flag}" ]]; then
        duplicate=1
      fi
    done
    if [[ "${duplicate}" -eq 0 ]]; then
      arch_flags+=("${flag}")
    fi
  fi
done < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)

if [[ "${#arch_flags[@]}" -eq 0 ]]; then
  arch_flags+=("-arch=native")
fi

# 每次启动都用当前环境的 CUDA 重新编译，避免使用历史遗留的预编译二进制
# （例如 tar 包解压时保留了打包时间戳，源码时间戳不比二进制新，导致换 CUDA
# 版本或换机器后仍误用旧二进制）。
# 多卡调度与报告生成已内置进二进制（无需 Python）；NVML 用于遥测采集。
echo "Building CUDA benchmark with $(nvcc --version | grep -i release)..."
nvcc -O3 -std=c++17 -lineinfo \
  "${arch_flags[@]}" \
  "${SRC_FILE}" \
  -lcublasLt -lcublas -lnvidia-ml \
  -o "${BIN_FILE}"

exec "${BIN_FILE}" "$@"
