# Dense NVIDIA GPU FLOPS Benchmark

一键测试 NVIDIA GPU 服务器的稠密 GEMM 吞吐量。

> 🤖 **本项目的代码主要由 AI（GLM-5.2）完成。**

## 测试过的 GPU

以下为本工具实际运行验证过的 GPU 列表（持续更新，后续测试新卡会追加）。

| GPU | 架构 | 计算能力 | 显存 | 备注 |
|---|---|---|---|---|
| NVIDIA GeForce RTX 5090 D v2 | Blackwell | 12.0 | 24 GB | NVFP4 / FP8 E4M3 / INT8 等均可用；FP8 E5M2 与 INT4 报 UNSUPPORTED |
| NVIDIA GeForce RTX 5090 | Blackwell | 12.0 | 32 GB | 性能数据见下表；FP8 E5M2 与 INT4 报 UNSUPPORTED |

> 欢迎补充：在你自己的卡上跑过后，把结果（精度、吞吐、是否 UNSUPPORTED）整理进此表即可。

## 实测性能数据

> 以下为实际运行本工具测得的单卡吞吐量（dense GEMM，非稀疏，isolated 隔离测试）。
> 取各精度最佳有效矩阵尺寸的吞吐。完整原始结果见 `results/`。
> 不同批次/驱动/功耗墙下数值可能有差异，仅供参考。

### RTX 5090（CC 12.0，驱动 580.105.08，CUDA 13.0）

| 精度 | 单卡吞吐 | 单位 |
|---|---|---|
| FP64 | 1.77 | TFLOPS |
| FP32 | 83.48 | TFLOPS |
| TF32 | 125.37 | TFLOPS |
| BF16 | 252.86 | TFLOPS |
| FP16 | 253.11 | TFLOPS |
| INT16 | 10.30 | TOPS |
| INT8 | 905.65 | TOPS |
| FP8 E4M3 | 862.81 | TFLOPS |
| NVFP4 | 1617.23 | TFLOPS |
| FP8 E5M2 | UNSUPPORTED | — |
| INT4 | UNSUPPORTED | — |

- INT4 / FP8 E5M2 在该架构下 cuBLASLt 无稠密 GEMM 内核，报 UNSUPPORTED，不伪造数值。

### RTX 5090 D v2（CC 12.0，驱动 580.105.08，CUDA 13.0）

| 精度 | 单卡吞吐 | 单位 |
|---|---|---|
| FP64 | 1.68 | TFLOPS |
| FP32 | 82.26 | TFLOPS |
| TF32 | 119.04 | TFLOPS |
| BF16 | 240.22 | TFLOPS |
| FP16 | 240.46 | TFLOPS |
| INT16 | 9.88 | TOPS |
| INT8 | 651.64 | TOPS |
| FP8 E4M3 | 613.65 | TFLOPS |
| NVFP4 | 1195.70 | TFLOPS |
| FP8 E5M2 | UNSUPPORTED | — |
| INT4 | UNSUPPORTED | — |

- INT4 / FP8 E5M2 在该架构下 cuBLASLt 无稠密 GEMM 内核，报 UNSUPPORTED，不伪造数值。
- 本次测试时机箱内其它卡有负载，INT8/FP8 最佳尺寸落在 4096³（受 PCIe/供电共享影响），数值偏低属正常环境差异。


## 支持的测试

| 名称 | 输入 | 累加/输出 | 单位 | 路径 |
|---|---|---|---|---|
| FP64 | FP64 | FP64 | TFLOPS | cuBLASLt |
| FP32 | FP32 | FP32 | TFLOPS | cuBLASLt PEDANTIC，严格 FP32 |
| TF32 | FP32 | FP32 | TFLOPS | cuBLASLt FAST_TF32 |
| BF16 | BF16 | FP32/BF16 | TFLOPS | cuBLASLt Tensor Core |
| FP16 | FP16 | FP32/FP16 | TFLOPS | cuBLASLt Tensor Core |
| INT16 | INT16 | INT32 | TOPS | 自定义 16x16 tiled CUDA Core GEMM |
| INT8 | INT8 | INT32 | TOPS | cuBLASLt Tensor Core |
| FP8 E4M3 | FP8 | FP32/FP16 | TFLOPS | cuBLASLt，TN 布局、Ada/Hopper+ |
| FP8 E5M2 | FP8 | FP32/FP16 | TFLOPS | cuBLASLt，TN 布局、Ada/Hopper+ |
| NVFP4 | FP4 E2M1 | FP32/BF16 | TFLOPS | cuBLASLt，VEC16_UE4M3 块缩放、Blackwell+ |
| INT4 | INT4 | — | TOPS | 不支持：cuBLASLt 无 INT4 稠密 GEMM 内核 |

所有测试均为稠密 GEMM，不使用 2:4 稀疏矩阵或稀疏 GEMM API。

INT16 是自定义 CUDA Core 内核，不是 Tensor Core 指标，不应直接与 INT8
Tensor Core TOPS 比较。

INT4 报告 UNSUPPORTED：cuBLASLt 不提供 INT4 稠密 GEMM 内核（`CUDA_R_4I` 在各
compute/output 组合与布局下均返回 `NOT_SUPPORTED`，已在多架构实测确认）。
Blackwell 等硬件虽支持 INT4 Tensor Core，但仅通过推理框架（CUTLASS /
TensorRT-LLM 等）以分组量化路径使用，而非标准稠密 GEMM API。量化推理（如
AWQ-INT4）的实际计算是 INT4 权重反量化回 FP16/BF16 再做 GEMM，并非纯 INT4
Tensor Core 稠密乘加，故本工具不伪造 INT4 数值。

NVFP4 使用 cuBLASLt 的 VEC16_UE4M3 块缩放路径（每 16 个 FP4 元素共享一个
UE4M3 缩放因子），需要 Blackwell 架构（CC 10.0+，如 RTX 50 系、B200）。A/B 为
FP4 E2M1（2 元素打包进 1 字节），C/D 输出 BF16，FP32 累加。测试中缩放因子固定
为 1.0 以便数值校验，实际应用里缩放因子通常逐块变化。

FP8（E4M3/E5M2）与 INT8 走 cuBLASLt 的 TN 布局、per-tensor 缩放路径（不设块
缩放属性，缩放因子为 1）。报告中每条结果的 `algorithm` 字段标注了所用算法 ID、
布局（TN/NN）和缩放路径（per-tensor scale / block-scaled），便于跨架构对比。
不同架构/GPU 的 FP8 算法可能不同：例如 Blackwell（RTX 50 系）上 E4M3 可用、
E5M2 无 cuBLASLt 内核（报 UNSUPPORTED）；Hopper 等其他架构的 FP8 支持情况以
实际运行结果为准。某精度在某卡上无 cuBLASLt 内核时，程序会报告 UNSUPPORTED
而非伪造结果。

## 环境要求

- Linux x86_64 / Arm64，或 Windows x86_64
- NVIDIA 驱动及 `nvidia-smi`
- CUDA Toolkit 13.0 或更新版本（含 cuBLASLt 与 NVML 库）
- Blackwell 架构（CC 10.0+）以使用 NVFP4
- 至少一张 NVIDIA CUDA GPU

本项目统一要求 CUDA Toolkit 13.0 或更新版本。多卡调度、NVML 遥测采集
（温度/功耗/时钟/利用率）与报告生成（md/csv/json）均已内置进二进制，
**无需安装 Python**。编译时链接 NVML 库（`-lnvidia-ml`，CUDA Toolkit 自带）；
NVML 不可用时遥测字段自动留空，不影响主测试流程。FP8 与 NVFP4 需要 GPU 和
CUDA/cuBLASLt 同时支持；程序会探测计算能力，不支持时报告 `UNSUPPORTED`，
不会伪造结果。NVFP4 另需 Blackwell 硬件（CC 10.0+）与 CUDA 13.0+ 编译环境
同时满足。

> `tools/run_multi_gpu.py` 为早期 Python 编排脚本，已保留作备用；默认流程
> 不再依赖它。如需使用，仍需 Python 3.8+。

FP8 和 INT8 的 cuBLASLt 调用使用 `TN` 布局（A 转置、B 不转置），这是
CUDA 13.0+ 对应高性能内核的必要条件。

## 一键运行

### Linux / macOS

```bash
tar -xzf nv_gpu_flops_test.tar.gz
cd nv_gpu_flops_test
./run_gpu_flops.sh
```

如果文件系统没有保留可执行权限：

```bash
bash run_gpu_flops.sh
```

### Windows

在「开发者命令提示符」或已配置好 CUDA 环境变量的命令提示符（CMD）中：

```cmd
cd nv_gpu_flops_test
run_gpu_flops.bat
```

脚本会自动探测 GPU 计算能力、用 `nvcc` 编译，再直接运行二进制生成报告
（多卡调度与报告生成已内置，无需 Python）。


## 常用选项

```bash
# 快速测试
./run_gpu_flops.sh --quick

# 只测 GPU 0 和 GPU 1
./run_gpu_flops.sh --gpus 0,1

# 只测部分精度
./run_gpu_flops.sh --precisions int8,fp8_e4m3,fp16,bf16,tf32,fp32

# 只测单卡隔离性能
./run_gpu_flops.sh --mode isolated

# 只测多卡并发性能
./run_gpu_flops.sh --mode concurrent

# 只测单张 GPU（--device 优先于 --gpus）
./run_gpu_flops.sh --device 0

# 自定义迭代、预热和 workspace
./run_gpu_flops.sh --iters 50 --warmup 15 --workspace-mb 512

# 指定结果目录
./run_gpu_flops.sh --output /data/gpu-benchmark-result
```

Windows 上把 `./run_gpu_flops.sh` 换成 `run_gpu_flops.bat`，参数完全一致：

```cmd
run_gpu_flops.bat --quick
run_gpu_flops.bat --gpus 0,1 --mode isolated
run_gpu_flops.bat --output D:\gpu-benchmark-result
```

## 输出

默认写入：

```text
results/YYYYMMDDTHHMMSSZ/
├── isolated_gpu0.json
├── concurrent_gpu0.json
├── report.md
├── report.csv
└── report.json
```

- `report.md`：适合直接阅读，只显示每种精度的最佳有效矩阵尺寸。
- `report.csv`：包含所有尺寸，适合导入 Excel。
- `report.json`：完整机器可读数据，包括配置、GPU 信息和原始结果。

## 测量口径

- GEMM 运算量为 `2 × M × N × K`。
- 浮点结果单位为 TFLOPS，整数结果单位为 TOPS。
- 每个尺寸先预热，再使用 CUDA Event 计时。
- 报告中位耗时和 P90 耗时。
- 输入使用固定值，计时完成后抽样校验输出；校验失败的结果不会标为最佳。
- `isolated` 模式逐卡执行，反映单卡隔离性能。
- `concurrent` 模式所有选定 GPU 同时执行，报告可相加的整机并发吞吐。

## 注意事项

1. 测试前应确保 GPU 没有其他重负载。
2. 功耗上限、散热、时钟锁定和 MIG 都会影响结果。
3. 脚本不会自动修改功耗或锁定时钟。
4. cuBLASLt 会根据 CUDA 版本、GPU 架构和矩阵形状选择算法，因此不同环境的
   最优尺寸可能不同。
5. `FP32` 使用 pedantic 计算模式以避免混入 TF32；`TF32` 是独立测试项。

## License

本项目基于 [MIT License](LICENSE) 开源，可自由使用、修改和分发。
