Dense GPU GEMM Benchmark
Generated: 2026-07-31T08:41:56.744+00:00
Driver: 580.105.08
CUDA compiler: Cuda compilation tools, release 13.0, V13.0.88
Dense only: no 2:4 sparsity or sparse GEMM APIs

[isolated]
GPU 0: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       261.386               1.77 TFLOPS   PASS
  fp32           8192       13.257               82.94 TFLOPS   PASS
  tf32           12288      29.599              125.37 TFLOPS   PASS
  bf16           16384      34.786              252.86 TFLOPS   PASS
  fp16           16384      34.752              253.11 TFLOPS   PASS
  int16          4096       13.344               10.30 TOPS     PASS
  int8           16384      9.712               905.65 TOPS     PASS
  fp8_e4m3       8192       1.274               862.81 TFLOPS   PASS
  nvfp4          8192       0.680              1617.23 TFLOPS   PASS

GPU 1: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       263.568               1.76 TFLOPS   PASS
  fp32           8192       13.245               83.01 TFLOPS   PASS
  tf32           12288      29.676              125.05 TFLOPS   PASS
  bf16           16384      34.898              252.05 TFLOPS   PASS
  fp16           16384      34.872              252.24 TFLOPS   PASS
  int16          4096       13.322               10.32 TOPS     PASS
  int8           16384      9.649               911.60 TOPS     PASS
  fp8_e4m3       16384      18.413              477.70 TFLOPS   PASS
  nvfp4          16384      5.343              1646.30 TFLOPS   PASS

GPU 2: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       264.035               1.76 TFLOPS   PASS
  fp32           8192       13.264               82.90 TFLOPS   PASS
  tf32           12288      29.737              124.79 TFLOPS   PASS
  bf16           16384      34.997              251.34 TFLOPS   PASS
  fp16           16384      34.962              251.59 TFLOPS   PASS
  int16          4096       13.483               10.19 TOPS     PASS
  int8           16384      9.665               910.08 TOPS     PASS
  fp8_e4m3       16384      18.463              476.41 TFLOPS   PASS
  nvfp4          8192       0.684              1607.02 TFLOPS   PASS

GPU 3: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       263.314               1.76 TFLOPS   PASS
  fp32           8192       13.218               83.18 TFLOPS   PASS
  tf32           12288      29.704              124.93 TFLOPS   PASS
  bf16           16384      34.945              251.71 TFLOPS   PASS
  fp16           16384      34.920              251.90 TFLOPS   PASS
  int16          4096       13.317               10.32 TOPS     PASS
  int8           16384      9.648               911.70 TOPS     PASS
  fp8_e4m3       16384      18.335              479.75 TFLOPS   PASS
  nvfp4          8192       0.678              1622.43 TFLOPS   PASS

GPU 4: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       264.133               1.76 TFLOPS   PASS
  fp32           8192       13.115               83.84 TFLOPS   PASS
  tf32           12288      29.638              125.21 TFLOPS   PASS
  bf16           16384      34.862              252.31 TFLOPS   PASS
  fp16           16384      34.841              252.47 TFLOPS   PASS
  int16          2048       1.669                10.29 TOPS     PASS
  int8           16384      9.490               926.92 TOPS     PASS
  fp8_e4m3       16384      18.373              478.74 TFLOPS   PASS
  nvfp4          8192       0.681              1614.42 TFLOPS   PASS

GPU 5: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       263.409               1.76 TFLOPS   PASS
  fp32           8192       13.259               82.92 TFLOPS   PASS
  tf32           12288      29.811              124.48 TFLOPS   PASS
  bf16           16384      34.819              252.62 TFLOPS   PASS
  fp16           16384      34.796              252.79 TFLOPS   PASS
  int16          4096       13.406               10.25 TOPS     PASS
  int8           16384      9.723               904.69 TOPS     PASS
  fp8_e4m3       16384      18.344              479.51 TFLOPS   PASS
  nvfp4          8192       0.683              1610.34 TFLOPS   PASS

GPU 6: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       262.807               1.77 TFLOPS   PASS
  fp32           8192       13.304               82.64 TFLOPS   PASS
  tf32           12288      29.574              125.48 TFLOPS   PASS
  bf16           16384      35.156              250.20 TFLOPS   PASS
  fp16           16384      35.151              250.24 TFLOPS   PASS
  int16          4096       13.334               10.31 TOPS     PASS
  int8           16384      9.756               901.57 TOPS     PASS
  fp8_e4m3       16384      18.205              483.16 TFLOPS   PASS
  nvfp4          8192       0.681              1613.82 TFLOPS   PASS

GPU 7: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       263.870               1.76 TFLOPS   PASS
  fp32           8192       13.191               83.35 TFLOPS   PASS
  tf32           12288      29.655              125.13 TFLOPS   PASS
  bf16           16384      34.890              252.11 TFLOPS   PASS
  fp16           16384      34.869              252.26 TFLOPS   PASS
  int16          4096       13.413               10.25 TOPS     PASS
  int8           16384      9.706               906.24 TOPS     PASS
  fp8_e4m3       16384      18.497              475.53 TFLOPS   PASS
  nvfp4          8192       0.685              1604.77 TFLOPS   PASS

[concurrent]
GPU 0: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       261.417               1.77 TFLOPS   PASS
  fp32           8192       13.213               83.22 TFLOPS   PASS
  tf32           12288      29.570              125.50 TFLOPS   PASS
  bf16           16384      34.782              252.89 TFLOPS   PASS
  fp16           16384      34.603              254.20 TFLOPS   PASS
  int16          4096       13.340               10.30 TOPS     PASS
  int8           16384      9.628               913.56 TOPS     PASS
  fp8_e4m3       8192       1.404               783.27 TFLOPS   PASS
  nvfp4          8192       0.681              1614.73 TFLOPS   PASS

GPU 1: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       263.555               1.76 TFLOPS   PASS
  fp32           8192       13.267               82.87 TFLOPS   PASS
  tf32           12288      29.678              125.04 TFLOPS   PASS
  bf16           16384      35.061              250.88 TFLOPS   PASS
  fp16           16384      34.858              252.34 TFLOPS   PASS
  int16          2048       1.679                10.23 TOPS     PASS
  int8           16384      9.652               911.36 TOPS     PASS
  fp8_e4m3       16384      18.410              477.78 TFLOPS   PASS
  nvfp4          8192       0.679              1618.22 TFLOPS   PASS

GPU 2: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       264.013               1.76 TFLOPS   PASS
  fp32           8192       13.330               82.49 TFLOPS   PASS
  tf32           12288      30.025              123.59 TFLOPS   PASS
  bf16           16384      35.173              250.08 TFLOPS   PASS
  fp16           16384      34.972              251.52 TFLOPS   PASS
  int16          4096       13.482               10.19 TOPS     PASS
  int8           16384      9.670               909.67 TOPS     PASS
  fp8_e4m3       16384      18.511              475.17 TFLOPS   PASS
  nvfp4          16384      5.384              1633.74 TFLOPS   PASS

GPU 3: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       263.316               1.76 TFLOPS   PASS
  fp32           8192       13.187               83.38 TFLOPS   PASS
  tf32           12288      29.706              124.92 TFLOPS   PASS
  bf16           16384      34.945              251.71 TFLOPS   PASS
  fp16           16384      34.908              251.98 TFLOPS   PASS
  int16          4096       13.315               10.32 TOPS     PASS
  int8           16384      9.650               911.53 TOPS     PASS
  fp8_e4m3       16384      18.340              479.62 TFLOPS   PASS
  nvfp4          8192       0.679              1619.75 TFLOPS   PASS

GPU 4: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       264.135               1.76 TFLOPS   PASS
  fp32           8192       13.248               82.99 TFLOPS   PASS
  tf32           12288      29.637              125.21 TFLOPS   PASS
  bf16           16384      35.017              251.19 TFLOPS   PASS
  fp16           16384      34.828              252.56 TFLOPS   PASS
  int16          4096       13.426               10.24 TOPS     PASS
  int8           16384      9.688               907.91 TOPS     PASS
  fp8_e4m3       16384      18.378              478.63 TFLOPS   PASS
  nvfp4          8192       0.685              1605.07 TFLOPS   PASS

GPU 5: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       263.211               1.76 TFLOPS   PASS
  fp32           8192       13.337               82.44 TFLOPS   PASS
  tf32           12288      29.813              124.47 TFLOPS   PASS
  bf16           16384      34.996              251.35 TFLOPS   PASS
  fp16           16384      34.968              251.55 TFLOPS   PASS
  int16          2048       1.675                10.26 TOPS     PASS
  int8           16384      9.722               904.80 TOPS     PASS
  fp8_e4m3       16384      18.529              474.72 TFLOPS   PASS
  nvfp4          8192       0.685              1605.00 TFLOPS   PASS

GPU 6: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       262.779               1.77 TFLOPS   PASS
  fp32           8192       13.273               82.84 TFLOPS   PASS
  tf32           12288      29.563              125.53 TFLOPS   PASS
  bf16           16384      35.298              249.20 TFLOPS   PASS
  fp16           16384      35.121              250.45 TFLOPS   PASS
  int16          4096       13.392               10.26 TOPS     PASS
  int8           16384      9.758               901.42 TOPS     PASS
  fp8_e4m3       16384      18.367              478.90 TFLOPS   PASS
  nvfp4          16384      5.399              1629.26 TFLOPS   PASS

GPU 7: NVIDIA GeForce RTX 5090
  Precision      M=N=K      Median(ms)      Throughput       Validation
  fp64           6144       263.633               1.76 TFLOPS   PASS
  fp32           8192       13.314               82.58 TFLOPS   PASS
  tf32           12288      29.735              124.80 TFLOPS   PASS
  bf16           16384      34.966              251.56 TFLOPS   PASS
  fp16           16384      34.940              251.75 TFLOPS   PASS
  int16          4096       13.444               10.22 TOPS     PASS
  int8           16384      9.747               902.46 TOPS     PASS
  fp8_e4m3       16384      18.550              474.19 TFLOPS   PASS
  nvfp4          8192       0.688              1599.24 TFLOPS   PASS

  Concurrent aggregate:
    fp64                14.10 TFLOPS
    fp32               662.81 TFLOPS
    tf32               999.05 TFLOPS
    bf16              2008.86 TFLOPS
    fp16              2016.34 TFLOPS
    int16               82.03 TOPS
    int8              7262.71 TOPS
    fp8_e4m3          4122.30 TFLOPS
    nvfp4            12925.01 TFLOPS

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

