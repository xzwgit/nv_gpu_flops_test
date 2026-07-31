Dense GPU GEMM Benchmark
Generated: 2026-07-31T11:28:09.324+00:00
Driver: 580.105.08
CUDA compiler: Cuda compilation tools, release 12.9, V12.9.41
Dense only: no 2:4 sparsity or sparse GEMM APIs

[isolated]
GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       257.062               1.80 TFLOPS   PASS
  fp32           8192       12.993               84.62 TFLOPS   PASS
  tf32           8192       4.894               224.67 TFLOPS   PASS
  bf16           16384      19.675              447.06 TFLOPS   PASS
  fp16           16384      20.849              421.90 TFLOPS   PASS
  int16          8192       178.056               6.18 TOPS     PASS
  int8           16384      10.405              845.40 TOPS     PASS
  fp8_e4m3       16384      10.140              867.48 TFLOPS   PASS
  nvfp4          16384      5.716              1538.90 TFLOPS   PASS

[UNSUPPORTED / SKIPPED]
  isolated GPU 0 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  isolated GPU 0 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture

