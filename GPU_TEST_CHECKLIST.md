# GPU 测试检查清单

> 运行 gpu_flops_test 前查看此清单，确保所有已测 GPU 的数据都更新到 README。
> 测完新 GPU 后，把结果追加到此清单 + 更新 README 表格。

## 已测 GPU 总览 (4 款)

| # | GPU | 架构 | CC | 显存 | 测试日期 | 结果位置 |
|---|---|---|---|---|---|---|
| 1 | RTX 5090 | Blackwell | 12.0 | 32 GB | 2026-07 | results/ (历史) + README |
| 2 | RTX 5090 D v2 | Blackwell | 12.0 | 24 GB | 2026-07 | results/ (历史) + README |
| 3 | RTX PRO 6000 Blackwell Server | Blackwell | 12.0 | 96 GB | 2026-07 | results/ (历史) + README |
| 4 | **RTX PRO 5000 Blackwell** | Blackwell | 12.0 | 48 GB | **2026-08-07** | **results/20260807T081713Z/** |

## 各 GPU 性能速查 (单卡最佳, TFLOPS)

| 精度 | 5090 (32G) | 5090D v2 (24G) | PRO 6000 (96G) | PRO 5000 (48G) |
|---|---|---|---|---|
| FP64 | 1.77 | 1.68 | 1.54 | **1.01** |
| FP32 | 83.24 | 83.70 | 83.60 | **52.41** |
| TF32 | 125.80 | 119.34 | 224.70 | **140.27** |
| BF16 | 254.03 | 240.72 | 457.61 | **257.12** |
| FP16 | 254.21 | 240.66 | 457.44 | **260.25** |
| INT8 | 906.14 | 665.27 | 845.39 | **551.88** |
| FP8 E4M3 | 773.52 | 660.66 | 905.32 | **557.34** |
| FP8 E5M2 | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| NVFP4 | 1617.46 | 1177.03 | 1608.21 | **1063.53** |
| INT4 | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |

## 测新 GPU 后的操作步骤 (别漏!)

1. [ ] 拉取结果到本机: `scp -r root@<GPU机>:/root/nv_gpu_flops_test/results/<timestamp> results/`
2. [ ] 提取各精度最佳值 (从 report.md)
3. [ ] **更新 README.md "测试过的 GPU"列表** (加一行)
4. [ ] **更新 README.md "实测性能数据"** (加一个表格)
5. [ ] **更新此清单** (加一行到上面的总览 + 速查表)
6. [ ] git commit + push

## 全部 Blackwell (CC 12.0) 通则

- FP8 E5M2: 全部 UNSUPPORTED (cuBLASLt 无稠密 GEMM 内核)
- INT4: 全部 UNSUPPORTED (同上)
- 其他精度: 全部 PASS (FP64/FP32/TF32/BF16/FP16/INT8/FP8 E4M3/NVFP4)
- CUDA: 统一 13.0
- 驱动: 统一 580.105.08
