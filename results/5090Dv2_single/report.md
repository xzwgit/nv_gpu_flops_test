Dense GPU GEMM Benchmark
Generated: 2026-07-31T11:40:18.651+00:00
Driver: 580.105.08
CUDA compiler: Cuda compilation tools, release 13.0, V13.0.88
Dense only: no 2:4 sparsity or sparse GEMM APIs

[isolated]
GPU 4: NVIDIA GeForce RTX 5090 D v2
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       276.301               1.68 TFLOPS   PASS
  fp32           8192       13.137               83.70 TFLOPS   PASS
  tf32           12288      31.095              119.34 TFLOPS   PASS
  bf16           16384      36.542              240.72 TFLOPS   PASS
  fp16           16384      36.550              240.66 TFLOPS   PASS
  int8           4096       0.207               665.27 TOPS     PASS
  fp8_e4m3       4096       0.208               660.66 TFLOPS   PASS
  nvfp4          4096       0.117              1177.03 TFLOPS   PASS

[UNSUPPORTED / SKIPPED]
  isolated GPU 4 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  isolated GPU 4 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture

