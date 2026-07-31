@echo off
REM ===========================================================================
REM run_gpu_flops.bat — Windows 版一键运行 (与 run_gpu_flops.sh 功能等价)
REM
REM 用途: 探测 GPU 计算能力, 用 nvcc 编译 src/gpu_dense_bench.cu, 再调用
REM       tools/run_multi_gpu.py 跑基准测试并生成 report.md/csv/json。
REM
REM 用法:
REM   run_gpu_flops.bat                  全量测试
REM   run_gpu_flops.bat --quick          快速测试
REM   run_gpu_flops.bat --gpus 0,1       只测 GPU 0 和 1
REM   run_gpu_flops.bat --precisions int8,fp8_e4m3,fp16
REM   run_gpu_flops.bat --mode isolated  只测单卡隔离性能
REM   run_gpu_flops.bat --output D:\bench\result
REM
REM 环境要求: NVIDIA 驱动 + nvidia-smi, CUDA Toolkit (nvcc), Python 3.8+
REM           编译运行需 NVIDIA GPU; CUDA Toolkit 13.0+ (见 README)
REM
REM 注: 末尾所有参数 (%*) 原样透传给 run_multi_gpu.py。
REM ===========================================================================

setlocal enabledelayedexpansion

REM --- 定位脚本所在目录 (批处理兼容写法) ---
set "ROOT_DIR=%~dp0"
REM 去掉末尾反斜杠, 统一路径风格
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "SRC_FILE=%ROOT_DIR%\src\gpu_dense_bench.cu"
set "BUILD_DIR=%ROOT_DIR%\build"
set "BIN_FILE=%BUILD_DIR%\gpu_dense_bench.exe"

REM --- 1. 依赖检查 ---
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvidia-smi not found. Install a working NVIDIA driver. 1>&2
    exit /b 1
)
where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found. Install the CUDA Toolkit and add its bin directory to PATH. 1>&2
    exit /b 1
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM --- 2. 探测每张 GPU 的计算能力, 生成 -gencode flags (去重) ---
REM     nvidia-smi 输出形如 "12.0", 去掉小数点变 "120" -> compute_120/sm_120
REM     但 CUDA gencode 用主版本号 (如 12.0 -> compute_120 不对, 应为 sm_120? )
REM     注意: .sh 里 cap="${cap//./}" 把 "12.0" -> "120", 与 .sh 保持一致。
set "ARCH_FLAGS="
set "SEEN="

REM 把 nvidia-smi 输出按行处理。用 for /f 解析。
REM nounits 格式: 每行一个 compute_cap, 如 "12.0"
for /f "usebackq tokens=*" %%C in (`nvidia-smi --query-gpu^=compute_cap --format^=csv,noheader,nounits 2^>nul`) do (
    set "CAP=%%C"
    REM 去除空白
    set "CAP=!CAP: =!"
    REM 去掉小数点: "12.0" -> "120"
    set "CAP=!CAP:.=!"
    REM 校验是否为纯数字
    set "ISNUM=1"
    for /f "delims=0123456789" %%X in ("!CAP!") do if not "%%X"=="" set "ISNUM=0"
    if "!ISNUM!"=="1" if not "!CAP!"=="" (
        set "FLAG=-gencode=arch=compute_!CAP!,code=sm_!CAP!"
        REM 去重: 检查 SEEN 是否已含此 flag
        set "DUP=0"
        if not "!SEEN!"=="" (
            for %%S in (!SEEN!) do if "%%S"=="!FLAG!" set "DUP=1"
        )
        if "!DUP!"=="0" (
            if defined ARCH_FLAGS (
                set "ARCH_FLAGS=!ARCH_FLAGS! !FLAG!"
            ) else (
                set "ARCH_FLAGS=!FLAG!"
            )
            set "SEEN=!SEEN! !FLAG!"
        )
    )
)

REM 无 GPU 计算能力信息时回退到 -arch=native
if not defined ARCH_FLAGS set "ARCH_FLAGS=-arch=native"

REM --- 3. 编译 ---
REM     每次 startup 都重新编译, 避免换 CUDA/换机器后误用旧二进制 (与 .sh 一致)。
REM     多卡调度与报告生成已内置进二进制(无需 Python); NVML 用于遥测采集。
echo Building CUDA benchmark...
nvcc --version 2>nul | findstr /i release >nul && for /f "delims=" %%V in ('nvcc --version ^| findstr /i release') do echo   %%V

nvcc -O3 -std=c++17 -lineinfo ^
    !ARCH_FLAGS! ^
    "%SRC_FILE%" ^
    -lcublasLt -lcublas -lnvidia-ml ^
    -o "%BIN_FILE%"

if errorlevel 1 (
    echo ERROR: nvcc compilation failed. 1>&2
    exit /b 1
)

REM --- 4. 运行二进制, 透传所有参数 ---
"%BIN_FILE%" %*
set "EXITCODE=%errorlevel%"

endlocal & exit /b %EXITCODE%
