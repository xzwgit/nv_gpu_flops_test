Dense GPU GEMM Benchmark
Generated: 2026-07-31T08:54:30.392+00:00
Driver: 580.105.08
CUDA compiler: Cuda compilation tools, release 13.0, V13.0.88
Dense only: no 2:4 sparsity or sparse GEMM APIs

[isolated]
GPU 4: NVIDIA GeForce RTX 5090 D v2
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       276.234               1.68 TFLOPS   PASS
  fp32           8192       13.366               82.26 TFLOPS   PASS
  tf32           12288      31.172              119.04 TFLOPS   PASS
  bf16           16384      36.617              240.22 TFLOPS   PASS
  fp16           16384      36.581              240.46 TFLOPS   PASS
  int16          4096       13.917                9.88 TOPS     PASS
  int8           4096       0.211               651.64 TOPS     PASS
  fp8_e4m3       4096       0.224               613.65 TFLOPS   PASS
  nvfp4          4096       0.115              1195.70 TFLOPS   PASS

[UNSUPPORTED / SKIPPED]
  isolated GPU 4 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  isolated GPU 4 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture

