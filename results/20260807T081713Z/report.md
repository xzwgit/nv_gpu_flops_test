Dense GPU GEMM Benchmark
Generated: 2026-08-07T08:24:40.317+00:00
Driver: 580.105.08
CUDA compiler: Cuda compilation tools, release 13.0, V13.0.88
Dense only: no 2:4 sparsity or sparse GEMM APIs

[isolated]
GPU 0: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       459.239               1.01 TFLOPS   PASS
  fp32           8192       20.978               52.41 TFLOPS   PASS
  tf32           12288      26.455              140.27 TFLOPS   PASS
  bf16           16384      34.288              256.53 TFLOPS   PASS
  fp16           16384      33.833              259.98 TFLOPS   PASS
  int8           16384      16.109              546.03 TOPS     PASS
  fp8_e4m3       8192       1.973               557.34 TFLOPS   PASS
  nvfp4          16384      8.386              1048.90 TFLOPS   PASS

GPU 1: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       471.294               0.98 TFLOPS   PASS
  fp32           8192       21.512               51.11 TFLOPS   PASS
  tf32           12288      27.143              136.71 TFLOPS   PASS
  bf16           16384      35.077              250.77 TFLOPS   PASS
  fp16           16384      34.696              253.52 TFLOPS   PASS
  int8           16384      16.574              530.72 TOPS     PASS
  fp8_e4m3       4096       0.340               404.80 TFLOPS   PASS
  nvfp4          16384      8.527              1031.50 TFLOPS   PASS

GPU 2: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       461.525               1.01 TFLOPS   PASS
  fp32           8192       21.324               51.56 TFLOPS   PASS
  tf32           12288      26.934              137.77 TFLOPS   PASS
  bf16           16384      34.888              252.13 TFLOPS   PASS
  fp16           16384      34.498              254.98 TFLOPS   PASS
  int8           16384      16.479              533.77 TOPS     PASS
  fp8_e4m3       4096       0.338               406.53 TFLOPS   PASS
  nvfp4          8192       1.063              1034.28 TFLOPS   PASS

GPU 3: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       465.560               1.00 TFLOPS   PASS
  fp32           8192       21.094               52.12 TFLOPS   PASS
  tf32           12288      26.541              139.82 TFLOPS   PASS
  bf16           16384      34.289              256.53 TFLOPS   PASS
  fp16           16384      33.906              259.43 TFLOPS   PASS
  int8           16384      16.088              546.76 TOPS     PASS
  fp8_e4m3       4096       0.325               423.11 TFLOPS   PASS
  nvfp4          16384      8.271              1063.53 TFLOPS   PASS

GPU 4: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       473.170               0.98 TFLOPS   PASS
  fp32           8192       21.772               50.50 TFLOPS   PASS
  tf32           12288      27.628              134.32 TFLOPS   PASS
  bf16           16384      35.770              245.90 TFLOPS   PASS
  fp16           16384      35.389              248.55 TFLOPS   PASS
  int8           16384      16.910              520.16 TOPS     PASS
  fp8_e4m3       4096       0.346               397.35 TFLOPS   PASS
  nvfp4          8192       1.090              1008.98 TFLOPS   PASS

GPU 5: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       461.150               1.01 TFLOPS   PASS
  fp32           8192       20.942               52.50 TFLOPS   PASS
  tf32           12288      26.442              140.34 TFLOPS   PASS
  bf16           16384      34.210              257.12 TFLOPS   PASS
  fp16           16384      33.798              260.25 TFLOPS   PASS
  int8           16384      16.094              546.53 TOPS     PASS
  fp8_e4m3       4096       0.329               417.88 TFLOPS   PASS
  nvfp4          16384      8.292              1060.85 TFLOPS   PASS

GPU 6: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       462.520               1.00 TFLOPS   PASS
  fp32           8192       21.147               51.99 TFLOPS   PASS
  tf32           12288      26.744              138.75 TFLOPS   PASS
  bf16           16384      34.696              253.52 TFLOPS   PASS
  fp16           16384      34.258              256.76 TFLOPS   PASS
  int8           16384      16.326              538.77 TOPS     PASS
  fp8_e4m3       4096       0.329               417.43 TFLOPS   PASS
  nvfp4          16384      8.452              1040.73 TFLOPS   PASS

GPU 7: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       464.595               1.00 TFLOPS   PASS
  fp32           8192       21.166               51.95 TFLOPS   PASS
  tf32           12288      26.752              138.71 TFLOPS   PASS
  bf16           16384      34.615              254.11 TFLOPS   PASS
  fp16           16384      34.229              256.98 TFLOPS   PASS
  int8           16384      16.298              539.72 TOPS     PASS
  fp8_e4m3       4096       0.329               417.23 TFLOPS   PASS
  nvfp4          16384      8.444              1041.64 TFLOPS   PASS

[concurrent]
GPU 0: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       459.224               1.01 TFLOPS   PASS
  fp32           8192       21.024               52.30 TFLOPS   PASS
  tf32           12288      26.560              139.71 TFLOPS   PASS
  bf16           16384      34.407              255.65 TFLOPS   PASS
  fp16           16384      33.970              258.94 TFLOPS   PASS
  int8           16384      16.041              548.33 TOPS     PASS
  fp8_e4m3       8192       1.975               556.74 TFLOPS   PASS
  nvfp4          16384      8.426              1043.92 TFLOPS   PASS

GPU 1: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       471.243               0.98 TFLOPS   PASS
  fp32           8192       21.567               50.98 TFLOPS   PASS
  tf32           12288      27.232              136.27 TFLOPS   PASS
  bf16           16384      35.225              249.71 TFLOPS   PASS
  fp16           16384      34.795              252.80 TFLOPS   PASS
  int8           16384      16.482              533.68 TOPS     PASS
  fp8_e4m3       4096       0.332               413.49 TFLOPS   PASS
  nvfp4          16384      8.610              1021.59 TFLOPS   PASS

GPU 2: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       461.525               1.01 TFLOPS   PASS
  fp32           8192       21.314               51.59 TFLOPS   PASS
  tf32           12288      26.978              137.55 TFLOPS   PASS
  bf16           16384      34.896              252.07 TFLOPS   PASS
  fp16           16384      34.533              254.72 TFLOPS   PASS
  int8           16384      16.354              537.85 TOPS     PASS
  fp8_e4m3       4096       0.338               406.80 TFLOPS   PASS
  nvfp4          8192       1.066              1031.30 TFLOPS   PASS

GPU 3: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       465.524               1.00 TFLOPS   PASS
  fp32           8192       21.090               52.13 TFLOPS   PASS
  tf32           12288      26.529              139.88 TFLOPS   PASS
  bf16           16384      34.243              256.88 TFLOPS   PASS
  fp16           16384      33.908              259.41 TFLOPS   PASS
  int8           16384      15.938              551.88 TOPS     PASS
  fp8_e4m3       4096       0.326               421.90 TFLOPS   PASS
  nvfp4          16384      8.289              1061.14 TFLOPS   PASS

GPU 4: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       473.009               0.98 TFLOPS   PASS
  fp32           8192       21.790               50.46 TFLOPS   PASS
  tf32           12288      27.659              134.16 TFLOPS   PASS
  bf16           16384      35.809              245.64 TFLOPS   PASS
  fp16           16384      35.503              247.76 TFLOPS   PASS
  int8           16384      16.792              523.82 TOPS     PASS
  fp8_e4m3       4096       0.348               395.05 TFLOPS   PASS
  nvfp4          8192       1.095              1004.00 TFLOPS   PASS

GPU 5: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       460.945               1.01 TFLOPS   PASS
  fp32           8192       21.035               52.27 TFLOPS   PASS
  tf32           12288      26.568              139.68 TFLOPS   PASS
  bf16           16384      34.340              256.15 TFLOPS   PASS
  fp16           16384      33.898              259.48 TFLOPS   PASS
  int8           16384      16.026              548.88 TOPS     PASS
  fp8_e4m3       4096       0.329               418.21 TFLOPS   PASS
  nvfp4          16384      8.381              1049.49 TFLOPS   PASS

GPU 6: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       461.855               1.00 TFLOPS   PASS
  fp32           8192       21.305               51.61 TFLOPS   PASS
  tf32           12288      27.004              137.42 TFLOPS   PASS
  bf16           16384      34.844              252.44 TFLOPS   PASS
  fp16           16384      34.562              254.50 TFLOPS   PASS
  int8           16384      16.339              538.36 TOPS     PASS
  fp8_e4m3       4096       0.328               418.65 TFLOPS   PASS
  nvfp4          8192       1.066              1031.30 TFLOPS   PASS

GPU 7: NVIDIA RTX PRO 5000 Blackwell
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       464.767               1.00 TFLOPS   PASS
  fp32           8192       21.532               51.06 TFLOPS   PASS
  tf32           12288      27.307              135.89 TFLOPS   PASS
  bf16           16384      35.247              249.55 TFLOPS   PASS
  fp16           16384      34.900              252.03 TFLOPS   PASS
  int8           16384      16.535              531.97 TOPS     PASS
  fp8_e4m3       4096       0.343               401.17 TFLOPS   PASS
  nvfp4          8192       1.073              1025.08 TFLOPS   PASS

  Concurrent aggregate:
    fp64                 7.99 TFLOPS
    fp32               412.40 TFLOPS
    tf32              1100.57 TFLOPS
    bf16              2018.09 TFLOPS
    fp16              2039.65 TFLOPS
    int8              4314.78 TOPS
    fp8_e4m3          3432.02 TFLOPS
    nvfp4             8267.80 TFLOPS

[UNSUPPORTED / SKIPPED]
  isolated GPU 0 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  isolated GPU 0 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  isolated GPU 1 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  isolated GPU 1 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  isolated GPU 1 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  isolated GPU 2 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  isolated GPU 2 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  isolated GPU 2 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  isolated GPU 3 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  isolated GPU 3 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  isolated GPU 3 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  isolated GPU 4 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  isolated GPU 4 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  isolated GPU 4 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  isolated GPU 5 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  isolated GPU 5 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  isolated GPU 5 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  isolated GPU 6 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  isolated GPU 6 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  isolated GPU 6 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  isolated GPU 7 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  isolated GPU 7 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  isolated GPU 7 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  concurrent GPU 0 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  concurrent GPU 0 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  concurrent GPU 1 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  concurrent GPU 1 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  concurrent GPU 1 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  concurrent GPU 2 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  concurrent GPU 2 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  concurrent GPU 2 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  concurrent GPU 3 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  concurrent GPU 3 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  concurrent GPU 3 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  concurrent GPU 4 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  concurrent GPU 4 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  concurrent GPU 4 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  concurrent GPU 5 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  concurrent GPU 5 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  concurrent GPU 5 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  concurrent GPU 6 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  concurrent GPU 6 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  concurrent GPU 6 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture
  concurrent GPU 7 int4: UNSUPPORTED - cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor Core is used via inference frameworks (CUTLASS/quantized paths), not standard dense GEMM APIs
  concurrent GPU 7 fp8_e4m3: UNSUPPORTED - All cuBLASLt heuristic algorithms failed
  concurrent GPU 7 fp8_e5m2: UNSUPPORTED - No cuBLASLt algorithm for fp8_e5m2 (CUBLAS_STATUS_NOT_SUPPORTED); cuBLASLt did not provide a kernel for this precision/layout on this GPU architecture

