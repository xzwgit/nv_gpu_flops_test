Dense GPU GEMM Benchmark
Generated: 2026-08-01T03:36:06.619+00:00
Driver: 580.105.08
CUDA compiler: Cuda compilation tools, release 13.0, V13.0.88
Dense only: no 2:4 sparsity or sparse GEMM APIs

[isolated]
GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           4096       89.439                1.54 TFLOPS   PASS
  fp32           8192       13.153               83.60 TFLOPS   PASS
  tf32           8192       4.893               224.70 TFLOPS   PASS
  bf16           16384      19.222              457.61 TFLOPS   PASS
  fp16           16384      19.229              457.44 TFLOPS   PASS
  int8           16384      10.405              845.39 TOPS     PASS
  fp8_e4m3       16384      9.716               905.32 TFLOPS   PASS
  nvfp4          16384      5.470              1608.21 TFLOPS   PASS

[UNSUPPORTED / SKIPPED]
  isolated GPU 0 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  isolated GPU 0 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture

