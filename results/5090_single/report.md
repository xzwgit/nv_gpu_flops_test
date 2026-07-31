Dense GPU GEMM Benchmark
Generated: 2026-07-31T11:39:18.572+00:00
Driver: 580.105.08
CUDA compiler: Cuda compilation tools, release 13.0, V13.0.88
Dense only: no 2:4 sparsity or sparse GEMM APIs

[isolated]
GPU 0: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       261.470               1.77 TFLOPS   PASS
  fp32           8192       13.209               83.24 TFLOPS   PASS
  tf32           12288      29.497              125.80 TFLOPS   PASS
  bf16           16384      34.627              254.03 TFLOPS   PASS
  fp16           16384      34.602              254.21 TFLOPS   PASS
  int8           16384      9.707               906.14 TOPS     PASS
  fp8_e4m3       8192       1.421               773.52 TFLOPS   PASS
  nvfp4          8192       0.680              1617.46 TFLOPS   PASS

[UNSUPPORTED / SKIPPED]
  isolated GPU 0 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  isolated GPU 0 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture

