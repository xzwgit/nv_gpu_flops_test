#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// FP8 支持检测：不依赖 CUDA_VERSION（该宏仅在 cuda_runtime_api.h 中定义，
// 仅 #include <cuda_runtime.h> 时在部分 CUDA 版本下不可用，例如 CUDA 13.0
// 下会被判定为未定义，导致 FP8 被错误禁用）。改为直接探测 cuda_fp8.h 是否
// 存在，该头文件自 CUDA 11.8 起随 Toolkit 提供。
#if defined(__has_include)
#  if __has_include(<cuda_fp8.h>)
#    include <cuda_fp8.h>
#    define GPU_BENCH_HAS_FP8 1
#  else
#    define GPU_BENCH_HAS_FP8 0
#  endif
#else
#  if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#    include <cuda_fp8.h>
#    define GPU_BENCH_HAS_FP8 1
#  else
#    define GPU_BENCH_HAS_FP8 0
#  endif
#endif

// NVFP4 (E2M1) 支持检测：cuda_fp4.h 自 CUDA 12.8 起随 Toolkit 提供，配合
// cuBLASLt 的 VEC16_UE4M3 块缩放路径使用，需要 Blackwell (CC 10.0+) 硬件。
// cuda_fp8.h 同时提供 UE4M3 缩放因子的转换函数 __nv_cvt_float_to_fp8。
#if defined(__has_include)
#  if __has_include(<cuda_fp4.h>) && GPU_BENCH_HAS_FP8
#    include <cuda_fp4.h>
#    define GPU_BENCH_HAS_FP4 1
#  else
#    define GPU_BENCH_HAS_FP4 0
#  endif
#else
#  define GPU_BENCH_HAS_FP4 0
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

// NVML 遥测支持：用 __has_include 探测 nvml.h（CUDA Toolkit 自带，提供温度/
// 功耗/时钟/利用率读取）。不可用时则遥测功能降级为空（不影响主测试流程）。
#if defined(__has_include)
#  if __has_include(<nvml.h>)
#    include <nvml.h>
#    define GPU_BENCH_HAS_NVML 1
#  else
#    define GPU_BENCH_HAS_NVML 0
#  endif
#else
#  define GPU_BENCH_HAS_NVML 0
#endif

namespace {

struct Options {
  int device = -1;                 // -1 = 未指定，跑全部卡；>=0 = 只跑该卡
  std::string gpus = "all";        // "all" 或逗号分隔的卡号列表
  std::string mode = "both";       // both / isolated / concurrent
  int iters = 30;
  int warmup = 10;
  int workspace_mb = 256;
  int timeout = 1800;              // 单卡测试超时秒数（保留字段，当前未强制）
  bool quick = false;
  std::string output;              // 输出目录；空 = 默认 results/<时间戳>
  std::string precisions =
      "int4,int8,fp8_e4m3,fp8_e5m2,nvfp4,fp16,bf16,tf32,fp32,fp64";
};

struct PrecisionSpec {
  std::string name;
  std::string input_type;
  std::string accumulator_type;
  std::string output_type;
  std::string unit;
  cudaDataType_t a_type;
  cudaDataType_t b_type;
  cudaDataType_t c_type;
  cudaDataType_t d_type;
  cublasComputeType_t compute_type;
  cudaDataType_t scale_type;
  double input_value;
  int min_cc;
};

struct Result {
  std::string precision;
  std::string input_type;
  std::string accumulator_type;
  std::string output_type;
  std::string unit;
  int m = 0;
  int n = 0;
  int k = 0;
  double median_ms = 0.0;
  double p90_ms = 0.0;
  double throughput = 0.0;
  bool best = false;
  std::string validation = "NOT_RUN";
  std::string status = "ERROR";
  std::string algorithm;
  std::size_t workspace_bytes = 0;
  std::string message;
};

// 单卡完整报告：对应 Python run_one() 返回的字典。device 信息与 results 分离，
// 与 Python best_rows() 的增强逻辑（从 device 取 name/compute_capability）对齐。
struct DeviceReport {
  std::string mode;                       // "isolated" / "concurrent"
  int device_id = 0;
  std::string status = "OK";              // OK / ERROR / TIMEOUT
  std::string error;                       // 非 OK 时的信息
  bool has_device = false;                // 是否成功取到设备属性（OK 时为 true）
  int device_index = 0;
  std::string device_name = "unknown";
  std::string compute_capability = "unknown";
  unsigned long long total_memory_bytes = 0;
  int multiprocessor_count = 0;
  std::vector<Result> results;
  std::map<std::string, std::string> telemetry_before;
  std::map<std::string, std::string> telemetry_after;
};

// 环境信息：对应 Python query_driver() / query_gpus() 的采集结果。
struct GpuInventoryItem {
  int index = 0;
  std::string name;
  std::string uuid;
  std::string memory_total;               // nvidia-smi nounits 输出为字符串
  std::string compute_cap;                // 可能为空（旧驱动无此字段）
  std::string power_limit;
};
struct Environment {
  std::string driver_version = "unknown";
  std::string nvcc_version = "unknown";
  std::vector<GpuInventoryItem> gpu_inventory;
};

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char ch : value) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  return out.str();
}

std::string cuda_error(cudaError_t status) {
  return std::string(cudaGetErrorName(status)) + ": " + cudaGetErrorString(status);
}

std::string cublas_error(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "CUBLAS_STATUS_" + std::to_string(static_cast<int>(status));
  }
}

std::vector<std::string> split(const std::string& value, char delimiter) {
  std::vector<std::string> items;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, delimiter)) {
    if (!item.empty()) items.push_back(item);
  }
  return items;
}

std::string utc_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t stamp = std::chrono::system_clock::to_time_t(now);
  std::tm tm_value{};
#ifdef _WIN32
  gmtime_s(&tm_value, &stamp);
#else
  gmtime_r(&stamp, &tm_value);
#endif
  std::ostringstream out;
  out << std::put_time(&tm_value, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

// ISO 8601 带毫秒的时间戳（对齐 Python datetime.isoformat()，顶层报告用）。
std::string iso_timestamp_ms() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t stamp = std::chrono::system_clock::to_time_t(now);
  std::tm tm_value{};
#ifdef _WIN32
  gmtime_s(&tm_value, &stamp);
#else
  gmtime_r(&stamp, &tm_value);
#endif
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;
  std::ostringstream out;
  out << std::put_time(&tm_value, "%Y-%m-%dT%H:%M:%S");
  out << "." << std::setfill('0') << std::setw(3) << ms.count() << "+00:00";
  return out.str();
}

// 执行外部命令并捕获 stdout（用于 nvidia-smi / nvcc --version 等信息采集）。
// 跨平台：Windows 用 _popen/_pclose，POSIX 用 popen/pclose。
std::string capture_command(const std::string& command) {
  std::string result;
#ifdef _WIN32
  FILE* pipe = _popen(command.c_str(), "r");
#else
  FILE* pipe = popen(command.c_str(), "r");
#endif
  if (!pipe) return result;
  char buffer[256];
  while (std::fgets(buffer, sizeof(buffer), pipe)) {
    result += buffer;
  }
#ifdef _WIN32
  _pclose(pipe);
#else
  pclose(pipe);
#endif
  return result;
}

// NVML 遥测：对应 Python read_telemetry()。读取 5 项指标，键名与 nvidia-smi
// --query-gpu 的字段名一致（temperature.gpu/power.draw/clocks.sm/clocks.mem/
// utilization.gpu），值对齐 nvidia-smi nounits 输出单位（power.draw 为 W，其余
// 为原单位）。NVML 不可用或调用失败时返回空 map（与 Python return {} 一致）。
std::map<std::string, std::string> read_telemetry(int device) {
  std::map<std::string, std::string> telemetry;
#if GPU_BENCH_HAS_NVML
  static bool nvml_tried = false;
  static bool nvml_ok = false;
  // nvmlInit 全局只做一次（线程安全）。
  if (!nvml_tried) {
    nvml_tried = true;
    nvml_ok = (nvmlInit() == NVML_SUCCESS);
  }
  if (!nvml_ok) return telemetry;

  nvmlDevice_t handle;
  if (nvmlDeviceGetHandleByIndex(device, &handle) != NVML_SUCCESS) return telemetry;

  unsigned int value = 0;
  if (nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU, &value) == NVML_SUCCESS) {
    telemetry["temperature.gpu"] = std::to_string(value);
  }
  if (nvmlDeviceGetPowerUsage(handle, &value) == NVML_SUCCESS) {
    // NVML 返回 mW，对齐 nvidia-smi nounits 的 W 输出。
    std::ostringstream s;
    s << std::fixed << std::setprecision(2) << (static_cast<double>(value) / 1000.0);
    telemetry["power.draw"] = s.str();
  }
  if (nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM, &value) == NVML_SUCCESS) {
    telemetry["clocks.sm"] = std::to_string(value);
  }
  if (nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM, &value) == NVML_SUCCESS) {
    telemetry["clocks.mem"] = std::to_string(value);
  }
  nvmlUtilization_t util{};
  if (nvmlDeviceGetUtilizationRates(handle, &util) == NVML_SUCCESS) {
    telemetry["utilization.gpu"] = std::to_string(util.gpu);
  }
#else
  (void)device;
#endif
  return telemetry;
}

// 采集环境信息：driver_version、nvcc_version、gpu_inventory。
// 对应 Python query_driver() + query_gpus()，均通过 nvidia-smi/nvcc 子进程获取，
// 与 Python 同源以保证字段一致。
Environment query_environment() {
  Environment env;
  // driver_version：nvidia-smi --query-gpu=driver_version 第一行。
  std::string driver_out = capture_command(
      "nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>nul");
  // POSIX 上 2>nul 无害（shell 忽略），为跨平台统一这里再尝试无重定向版兜底。
  if (driver_out.empty()) {
    driver_out = capture_command(
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits");
  }
  if (!driver_out.empty()) {
    std::istringstream stream(driver_out);
    std::getline(stream, env.driver_version);
    // 去除首尾空白。
    while (!env.driver_version.empty() &&
           (env.driver_version.back() == '\r' || env.driver_version.back() == '\n' ||
            env.driver_version.back() == ' ')) {
      env.driver_version.pop_back();
    }
  }

  // nvcc_version：nvcc --version 中含 "release" 的行。
  std::string nvcc_out = capture_command("nvcc --version");
  for (const std::string& line : split(nvcc_out, '\n')) {
    std::string trimmed = line;
    while (!trimmed.empty() && (trimmed.front() == ' ' || trimmed.front() == '\t' ||
                                trimmed.front() == '\r')) {
      trimmed.erase(trimmed.begin());
    }
    // 不区分大小写找 "release"。
    std::string lower = trimmed;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.find("release") != std::string::npos) {
      env.nvcc_version = trimmed;
      break;
    }
  }

  // gpu_inventory：nvidia-smi --query-gpu=index,name,uuid,memory.total,compute_cap,
  // power.limit。旧驱动无 compute_cap 时回退到不含它的字段集（与 Python 一致）。
  auto parse_inventory = [](const std::string& out,
                            bool has_compute_cap) -> std::vector<GpuInventoryItem> {
    std::vector<GpuInventoryItem> items;
    std::istringstream stream(out);
    std::string line;
    while (std::getline(stream, line)) {
      while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
        line.pop_back();
      }
      if (line.empty()) continue;
      std::vector<std::string> values = split(line, ',');
      for (auto& v : values) {
        size_t start = v.find_first_not_of(" \t");
        size_t end = v.find_last_not_of(" \t\r\n");
        v = (start == std::string::npos) ? std::string() : v.substr(start, end - start + 1);
      }
      if (values.size() < (has_compute_cap ? 6u : 5u)) continue;
      GpuInventoryItem item;
      try {
        item.index = std::stoi(values[0]);
      } catch (...) {
        continue;
      }
      item.name = values[1];
      item.uuid = values[2];
      item.memory_total = values[3];
      if (has_compute_cap) {
        item.compute_cap = values[4];
        item.power_limit = values[5];
      } else {
        item.power_limit = values[4];
      }
      items.push_back(item);
    }
    return items;
  };

  std::string inv_out = capture_command(
      "nvidia-smi --query-gpu=index,name,uuid,memory.total,compute_cap,power.limit "
      "--format=csv,noheader,nounits");
  if (!inv_out.empty()) {
    env.gpu_inventory = parse_inventory(inv_out, true);
  } else {
    // 回退：旧驱动无 compute_cap。
    std::string inv_fallback = capture_command(
        "nvidia-smi --query-gpu=index,name,uuid,memory.total,power.limit "
        "--format=csv,noheader,nounits");
    env.gpu_inventory = parse_inventory(inv_fallback, false);
  }
  return env;
}

std::size_t type_size(cudaDataType_t type) {
  switch (type) {
    case CUDA_R_8I: return 1;
    case CUDA_R_16F: return 2;
    case CUDA_R_16BF: return 2;
    case CUDA_R_32I: return 4;
    case CUDA_R_32F: return 4;
    case CUDA_R_64F: return 8;
#if GPU_BENCH_HAS_FP8
    case CUDA_R_8F_E4M3: return 1;
    case CUDA_R_8F_E5M2: return 1;
#endif
    default: return 0;
  }
}

template <typename T>
__device__ T convert_from_float(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __half convert_from_float<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __nv_bfloat16 convert_from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

#if GPU_BENCH_HAS_FP8
template <>
__device__ __nv_fp8_e4m3 convert_from_float<__nv_fp8_e4m3>(float value) {
  return __nv_fp8_e4m3(value);
}

template <>
__device__ __nv_fp8_e5m2 convert_from_float<__nv_fp8_e5m2>(float value) {
  return __nv_fp8_e5m2(value);
}
#endif

template <typename T>
__global__ void fill_kernel(T* pointer, std::size_t count, float value) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) pointer[index] = convert_from_float<T>(value);
}

cudaError_t fill_buffer(
    void* pointer, std::size_t count, cudaDataType_t type, float value,
    cudaStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  switch (type) {
    case CUDA_R_8I:
      fill_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<std::int8_t*>(pointer), count, value);
      break;
    case CUDA_R_16F:
      fill_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<__half*>(pointer), count, value);
      break;
    case CUDA_R_16BF:
      fill_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<__nv_bfloat16*>(pointer), count, value);
      break;
    case CUDA_R_32I:
      fill_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<std::int32_t*>(pointer), count, value);
      break;
    case CUDA_R_32F:
      fill_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<float*>(pointer), count, value);
      break;
    case CUDA_R_64F:
      fill_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<double*>(pointer), count, value);
      break;
#if GPU_BENCH_HAS_FP8
    case CUDA_R_8F_E4M3:
      fill_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<__nv_fp8_e4m3*>(pointer), count, value);
      break;
    case CUDA_R_8F_E5M2:
      fill_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<__nv_fp8_e5m2*>(pointer), count, value);
      break;
#endif
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

#if GPU_BENCH_HAS_FP4
// FP4 (E2M1) 每两个元素打包进 1 字节（__nv_fp4x2_storage_t == unsigned char）。
// 方阵 size×size 共 size*size 个逻辑元素，对应 size*size/2 个打包字节。
// 用 __nv_cvt_float2_to_fp4x2 把一对 float 值转成一个打包字节（返回 storage 类型）。
__global__ void fill_fp4_kernel(
    __nv_fp4x2_storage_t* pointer, std::size_t pair_count, float value) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < pair_count) {
    pointer[index] = __nv_cvt_float2_to_fp4x2(
        make_float2(value, value), __NV_E2M1, cudaRoundNearest);
  }
}

// roundoff(x, m)：将 x 向上取整到 m 的倍数（m 为 2 的幂或任意正整数）。
inline std::size_t roundoff(std::size_t x, std::size_t m) {
  return ((x + m - 1) / m) * m;
}

// NVFP4 块缩放张量（VEC16_UE4M3）大小，单位为字节（每 16 个数据元素 1 个 UE4M3）。
// 布局公式来自 NVIDIA cuBLASLt 文档与官方 LtNvfp4Matmul 示例 helpers.h：
//   S_VSCALE = 16，BLOCK_ROWS = 4 * 16 = 64，BLOCK_COLS = 32 * 4 = 128
//   s_rows = roundoff(inner, 64) / 16
//   s_cols = roundoff(outer, 128)
//   字节数 = s_rows * s_cols
// 对 size×size 方阵：inner = outer = size。
inline std::size_t nvfp4_scale_bytes(int size) {
  const std::size_t inner = static_cast<std::size_t>(size);
  const std::size_t outer = static_cast<std::size_t>(size);
  const std::size_t s_rows = roundoff(inner, 64) / 16;
  const std::size_t s_cols = roundoff(outer, 128);
  return s_rows * s_cols;
}
#endif  // GPU_BENCH_HAS_FP4

double percentile(std::vector<float> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double raw = std::ceil(p * static_cast<double>(values.size())) - 1.0;
  const std::size_t index = static_cast<std::size_t>(
      std::max(0.0, std::min(raw, static_cast<double>(values.size() - 1))));
  return static_cast<double>(values[index]);
}

bool validate_first_element(
    void* output, cudaDataType_t type, double expected, double relative_tolerance,
    std::string* message) {
  double actual = 0.0;
  cudaError_t status = cudaSuccess;
  switch (type) {
    case CUDA_R_16F: {
      __half value{};
      status = cudaMemcpy(&value, output, sizeof(value), cudaMemcpyDeviceToHost);
      actual = __half2float(value);
      break;
    }
    case CUDA_R_16BF: {
      __nv_bfloat16 value{};
      status = cudaMemcpy(&value, output, sizeof(value), cudaMemcpyDeviceToHost);
      actual = __bfloat162float(value);
      break;
    }
    case CUDA_R_32I: {
      std::int32_t value = 0;
      status = cudaMemcpy(&value, output, sizeof(value), cudaMemcpyDeviceToHost);
      actual = static_cast<double>(value);
      break;
    }
    case CUDA_R_32F: {
      float value = 0.0f;
      status = cudaMemcpy(&value, output, sizeof(value), cudaMemcpyDeviceToHost);
      actual = static_cast<double>(value);
      break;
    }
    case CUDA_R_64F: {
      double value = 0.0;
      status = cudaMemcpy(&value, output, sizeof(value), cudaMemcpyDeviceToHost);
      actual = value;
      break;
    }
    default:
      *message = "Unsupported output type for validation";
      return false;
  }
  if (status != cudaSuccess) {
    *message = "Validation copy failed: " + cuda_error(status);
    return false;
  }
  const double tolerance =
      std::max(1.0e-6, std::abs(expected) * relative_tolerance);
  const bool ok = std::isfinite(actual) && std::abs(actual - expected) <= tolerance;
  std::ostringstream out;
  out << std::setprecision(10) << "actual=" << actual
      << ", expected=" << expected << ", tolerance=" << tolerance;
  *message = out.str();
  return ok;
}

const void* alpha_pointer(const PrecisionSpec& spec,
                          float* alpha_f, double* alpha_d, std::int32_t* alpha_i) {
  if (spec.scale_type == CUDA_R_64F) return alpha_d;
  if (spec.scale_type == CUDA_R_32I) return alpha_i;
  return alpha_f;
}

const void* beta_pointer(const PrecisionSpec& spec,
                         float* beta_f, double* beta_d, std::int32_t* beta_i) {
  if (spec.scale_type == CUDA_R_64F) return beta_d;
  if (spec.scale_type == CUDA_R_32I) return beta_i;
  return beta_f;
}

Result run_cublaslt_size(
    cublasLtHandle_t handle, cudaStream_t stream, const PrecisionSpec& spec,
    int size, int iters, int warmup, std::size_t requested_workspace) {
  Result result;
  result.precision = spec.name;
  result.input_type = spec.input_type;
  result.accumulator_type = spec.accumulator_type;
  result.output_type = spec.output_type;
  result.unit = spec.unit;
  result.m = result.n = result.k = size;

  void* a = nullptr;
  void* b = nullptr;
  void* d = nullptr;
  void* workspace = nullptr;
  std::size_t workspace_bytes = requested_workspace;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatrixLayout_t d_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;

  auto cleanup = [&]() {
    if (start) cudaEventDestroy(start);
    if (end) cudaEventDestroy(end);
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
    if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (workspace) cudaFree(workspace);
    if (a) cudaFree(a);
    if (b) cudaFree(b);
    if (d) cudaFree(d);
  };

  const std::size_t a_bytes =
      static_cast<std::size_t>(size) * size * type_size(spec.a_type);
  const std::size_t b_bytes =
      static_cast<std::size_t>(size) * size * type_size(spec.b_type);
  const std::size_t d_bytes =
      static_cast<std::size_t>(size) * size * type_size(spec.d_type);
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (cuda_status != cudaSuccess) {
    result.message = "cudaMemGetInfo failed: " + cuda_error(cuda_status);
    cleanup();
    return result;
  }
  const long double required = static_cast<long double>(a_bytes) + b_bytes + d_bytes;
  if (required + requested_workspace > static_cast<long double>(free_bytes) * 0.75L) {
    result.status = "SKIPPED";
    result.message = "Matrix and workspace would use more than 75% of free GPU memory";
    cleanup();
    return result;
  }

  if ((cuda_status = cudaMalloc(&a, a_bytes)) != cudaSuccess ||
      (cuda_status = cudaMalloc(&b, b_bytes)) != cudaSuccess ||
      (cuda_status = cudaMalloc(&d, d_bytes)) != cudaSuccess) {
    result.message = "Matrix allocation failed: " + cuda_error(cuda_status);
    cudaGetLastError();
    cleanup();
    return result;
  }
  if (workspace_bytes > 0) {
    cuda_status = cudaMalloc(&workspace, workspace_bytes);
    if (cuda_status != cudaSuccess) {
      cudaGetLastError();
      workspace = nullptr;
      workspace_bytes = 0;
    }
  }
  result.workspace_bytes = workspace_bytes;

  const float input_value = static_cast<float>(spec.input_value);
  if ((cuda_status = fill_buffer(
           a, static_cast<std::size_t>(size) * size, spec.a_type,
           input_value, stream)) != cudaSuccess ||
      (cuda_status = fill_buffer(
           b, static_cast<std::size_t>(size) * size, spec.b_type,
           input_value, stream)) != cudaSuccess ||
      (cuda_status = fill_buffer(
           d, static_cast<std::size_t>(size) * size, spec.d_type,
           0.0f, stream)) != cudaSuccess ||
      (cuda_status = cudaStreamSynchronize(stream)) != cudaSuccess) {
    result.message = "Buffer initialization failed: " + cuda_error(cuda_status);
    cleanup();
    return result;
  }

  cublasStatus_t blas_status =
      cublasLtMatmulDescCreate(&operation, spec.compute_type, spec.scale_type);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    result.message = "Matmul descriptor creation failed: " + cublas_error(blas_status);
    cleanup();
    return result;
  }
  // CUDA 11.8/12.x cuBLASLt FP8 kernels require TN layout. The regular-order
  // INT8 IMMA path has the same requirement. Matrices are square in this
  // benchmark, so transposing A does not change allocation dimensions.
  const bool requires_tn =
      spec.name == "int8" ||
      spec.name == "fp8_e4m3" ||
      spec.name == "fp8_e5m2";
  cublasOperation_t transpose_a =
      requires_tn ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transpose_b = CUBLAS_OP_N;
  if ((blas_status = cublasLtMatmulDescSetAttribute(
           operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_a,
           sizeof(transpose_a))) != CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatmulDescSetAttribute(
           operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_b,
           sizeof(transpose_b))) != CUBLAS_STATUS_SUCCESS) {
    result.message = "Matmul descriptor configuration failed: " +
                     cublas_error(blas_status);
    cleanup();
    return result;
  }

  if ((blas_status = cublasLtMatrixLayoutCreate(
           &a_layout, spec.a_type, size, size, size)) != CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatrixLayoutCreate(
           &b_layout, spec.b_type, size, size, size)) != CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatrixLayoutCreate(
           &c_layout, spec.c_type, size, size, size)) != CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatrixLayoutCreate(
           &d_layout, spec.d_type, size, size, size)) != CUBLAS_STATUS_SUCCESS) {
    result.message = "Matrix layout creation failed: " + cublas_error(blas_status);
    cleanup();
    return result;
  }
  if ((blas_status = cublasLtMatmulPreferenceCreate(&preference)) !=
          CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatmulPreferenceSetAttribute(
           preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
           &workspace_bytes, sizeof(workspace_bytes))) != CUBLAS_STATUS_SUCCESS) {
    result.message = "Preference configuration failed: " + cublas_error(blas_status);
    cleanup();
    return result;
  }

  constexpr int kRequestedAlgorithms = 16;
  std::vector<cublasLtMatmulHeuristicResult_t> heuristics(kRequestedAlgorithms);
  int returned = 0;
  blas_status = cublasLtMatmulAlgoGetHeuristic(
      handle, operation, a_layout, b_layout, c_layout, d_layout, preference,
      kRequestedAlgorithms, heuristics.data(), &returned);
  if (blas_status != CUBLAS_STATUS_SUCCESS || returned == 0) {
    result.status = "UNSUPPORTED";
    result.message = "No cuBLASLt algorithm for " + spec.name + " (" +
                     cublas_error(blas_status) +
                     "); cuBLASLt did not provide a kernel for this "
                     "precision/layout on this GPU architecture";
    cleanup();
    return result;
  }

  float alpha_f = 1.0f;
  float beta_f = 0.0f;
  double alpha_d = 1.0;
  double beta_d = 0.0;
  std::int32_t alpha_i = 1;
  std::int32_t beta_i = 0;
  const void* alpha = alpha_pointer(spec, &alpha_f, &alpha_d, &alpha_i);
  const void* beta = beta_pointer(spec, &beta_f, &beta_d, &beta_i);

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  int best_index = -1;
  float best_trial_ms = std::numeric_limits<float>::infinity();
  const int trial_count = 2;
  for (int index = 0; index < returned; ++index) {
    if (heuristics[index].state != CUBLAS_STATUS_SUCCESS ||
        heuristics[index].workspaceSize > workspace_bytes) {
      continue;
    }
    blas_status = cublasLtMatmul(
        handle, operation, alpha, a, a_layout, b, b_layout, beta, d, c_layout,
        d, d_layout, &heuristics[index].algo, workspace, workspace_bytes, stream);
    if (blas_status != CUBLAS_STATUS_SUCCESS ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }
    cudaEventRecord(start, stream);
    bool trial_ok = true;
    for (int trial = 0; trial < trial_count; ++trial) {
      blas_status = cublasLtMatmul(
          handle, operation, alpha, a, a_layout, b, b_layout, beta, d, c_layout,
          d, d_layout, &heuristics[index].algo, workspace, workspace_bytes, stream);
      if (blas_status != CUBLAS_STATUS_SUCCESS) {
        trial_ok = false;
        break;
      }
    }
    if (!trial_ok) continue;
    cudaEventRecord(end, stream);
    if (cudaEventSynchronize(end) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, end);
    const float per_iteration = elapsed / trial_count;
    if (per_iteration < best_trial_ms) {
      best_trial_ms = per_iteration;
      best_index = index;
    }
  }
  if (best_index < 0) {
    result.status = "UNSUPPORTED";
    result.message = "All cuBLASLt heuristic algorithms failed";
    cleanup();
    return result;
  }

  const auto& selected = heuristics[best_index];
  int algorithm_id = -1;
  std::size_t attribute_written = 0;
  cublasLtMatmulAlgoConfigGetAttribute(
      &selected.algo, CUBLASLT_ALGO_CONFIG_ID, &algorithm_id,
      sizeof(algorithm_id), &attribute_written);
  result.algorithm = "cuBLASLt algo " + std::to_string(algorithm_id) +
                     (requires_tn ? " layout=TN" : " layout=NN") +
                     " per-tensor scale";
  result.workspace_bytes = selected.workspaceSize;

  for (int iteration = 0; iteration < warmup; ++iteration) {
    blas_status = cublasLtMatmul(
        handle, operation, alpha, a, a_layout, b, b_layout, beta, d, c_layout,
        d, d_layout, &selected.algo, workspace, workspace_bytes, stream);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
      result.message = "Warmup failed: " + cublas_error(blas_status);
      cleanup();
      return result;
    }
  }
  if ((cuda_status = cudaStreamSynchronize(stream)) != cudaSuccess) {
    result.message = "Warmup synchronization failed: " + cuda_error(cuda_status);
    cleanup();
    return result;
  }

  std::vector<float> times;
  times.reserve(iters);
  for (int iteration = 0; iteration < iters; ++iteration) {
    cudaEventRecord(start, stream);
    blas_status = cublasLtMatmul(
        handle, operation, alpha, a, a_layout, b, b_layout, beta, d, c_layout,
        d, d_layout, &selected.algo, workspace, workspace_bytes, stream);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
      result.message = "Timed matmul failed: " + cublas_error(blas_status);
      cleanup();
      return result;
    }
    cudaEventRecord(end, stream);
    if ((cuda_status = cudaEventSynchronize(end)) != cudaSuccess) {
      result.message = "Timed synchronization failed: " + cuda_error(cuda_status);
      cleanup();
      return result;
    }
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, end);
    times.push_back(elapsed);
  }

  result.median_ms = percentile(times, 0.5);
  result.p90_ms = percentile(times, 0.9);
  const long double operations =
      2.0L * static_cast<long double>(size) * size * size;
  result.throughput = static_cast<double>(
      operations / (static_cast<long double>(result.median_ms) / 1000.0L) /
      1.0e12L);

  const double expected =
      static_cast<double>(size) * spec.input_value * spec.input_value;
  const double tolerance =
      (spec.name == "fp64") ? 1.0e-10 :
      (spec.name == "fp32") ? 1.0e-5 :
      (spec.name == "int8") ? 0.0 : 0.03;
  std::string validation_message;
  const bool valid = validate_first_element(
      d, spec.d_type, expected, tolerance, &validation_message);
  result.validation = valid ? "PASS" : "FAIL";
  result.status = valid ? "OK" : "VALIDATION_FAILED";
  result.message = validation_message;
  cleanup();
  return result;
}

constexpr int kInt16Tile = 16;

__global__ void int16_gemm_kernel(
    const std::int16_t* a, const std::int16_t* b, std::int32_t* c,
    int m, int n, int k) {
  __shared__ std::int16_t a_tile[kInt16Tile][kInt16Tile];
  __shared__ std::int16_t b_tile[kInt16Tile][kInt16Tile];
  const int row = blockIdx.y * kInt16Tile + threadIdx.y;
  const int column = blockIdx.x * kInt16Tile + threadIdx.x;
  std::int32_t accumulator = 0;

  for (int base = 0; base < k; base += kInt16Tile) {
    const int a_column = base + threadIdx.x;
    const int b_row = base + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && a_column < k) ? a[row * k + a_column] : 0;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < k && column < n) ? b[b_row * n + column] : 0;
    __syncthreads();
#pragma unroll
    for (int inner = 0; inner < kInt16Tile; ++inner) {
      accumulator += static_cast<std::int32_t>(a_tile[threadIdx.y][inner]) *
                     static_cast<std::int32_t>(b_tile[inner][threadIdx.x]);
    }
    __syncthreads();
  }
  if (row < m && column < n) c[row * n + column] = accumulator;
}

Result run_int16_size(
    cudaStream_t stream, int size, int iters, int warmup) {
  Result result;
  result.precision = "int16";
  result.input_type = "INT16";
  result.accumulator_type = "INT32";
  result.output_type = "INT32";
  result.unit = "TOPS";
  result.m = result.n = result.k = size;
  result.algorithm = "custom tiled 16x16 CUDA Core GEMM";

  std::int16_t* a = nullptr;
  std::int16_t* b = nullptr;
  std::int32_t* c = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  const std::size_t element_count = static_cast<std::size_t>(size) * size;
  const std::size_t a_bytes = element_count * sizeof(std::int16_t);
  const std::size_t c_bytes = element_count * sizeof(std::int32_t);

  auto cleanup = [&]() {
    if (start) cudaEventDestroy(start);
    if (end) cudaEventDestroy(end);
    if (a) cudaFree(a);
    if (b) cudaFree(b);
    if (c) cudaFree(c);
  };
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess ||
      static_cast<long double>(2 * a_bytes + c_bytes) >
          static_cast<long double>(free_bytes) * 0.75L) {
    result.status = "SKIPPED";
    result.message = status == cudaSuccess
        ? "Matrices would use more than 75% of free GPU memory"
        : "cudaMemGetInfo failed: " + cuda_error(status);
    cleanup();
    return result;
  }
  if ((status = cudaMalloc(&a, a_bytes)) != cudaSuccess ||
      (status = cudaMalloc(&b, a_bytes)) != cudaSuccess ||
      (status = cudaMalloc(&c, c_bytes)) != cudaSuccess) {
    result.message = "INT16 allocation failed: " + cuda_error(status);
    cudaGetLastError();
    cleanup();
    return result;
  }
  const int fill_threads = 256;
  const int fill_blocks =
      static_cast<int>((element_count + fill_threads - 1) / fill_threads);
  fill_kernel<<<fill_blocks, fill_threads, 0, stream>>>(a, element_count, 1.0f);
  fill_kernel<<<fill_blocks, fill_threads, 0, stream>>>(b, element_count, 1.0f);
  fill_kernel<<<fill_blocks, fill_threads, 0, stream>>>(c, element_count, 0.0f);
  if ((status = cudaGetLastError()) != cudaSuccess ||
      (status = cudaStreamSynchronize(stream)) != cudaSuccess) {
    result.message = "INT16 initialization failed: " + cuda_error(status);
    cleanup();
    return result;
  }

  const dim3 block(kInt16Tile, kInt16Tile);
  const dim3 grid(
      (size + kInt16Tile - 1) / kInt16Tile,
      (size + kInt16Tile - 1) / kInt16Tile);
  for (int iteration = 0; iteration < warmup; ++iteration) {
    int16_gemm_kernel<<<grid, block, 0, stream>>>(a, b, c, size, size, size);
  }
  if ((status = cudaStreamSynchronize(stream)) != cudaSuccess) {
    result.message = "INT16 warmup failed: " + cuda_error(status);
    cleanup();
    return result;
  }
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  std::vector<float> times;
  times.reserve(iters);
  for (int iteration = 0; iteration < iters; ++iteration) {
    cudaEventRecord(start, stream);
    int16_gemm_kernel<<<grid, block, 0, stream>>>(a, b, c, size, size, size);
    cudaEventRecord(end, stream);
    if ((status = cudaEventSynchronize(end)) != cudaSuccess) {
      result.message = "INT16 timed kernel failed: " + cuda_error(status);
      cleanup();
      return result;
    }
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, end);
    times.push_back(elapsed);
  }
  result.median_ms = percentile(times, 0.5);
  result.p90_ms = percentile(times, 0.9);
  const long double operations =
      2.0L * static_cast<long double>(size) * size * size;
  result.throughput = static_cast<double>(
      operations / (static_cast<long double>(result.median_ms) / 1000.0L) /
      1.0e12L);
  std::string validation_message;
  const bool valid = validate_first_element(
      c, CUDA_R_32I, static_cast<double>(size), 0.0, &validation_message);
  result.validation = valid ? "PASS" : "FAIL";
  result.status = valid ? "OK" : "VALIDATION_FAILED";
  result.message = validation_message +
      "; custom CUDA Core result, not an INT16 Tensor Core specification";
  cleanup();
  return result;
}

#if GPU_BENCH_HAS_FP4
// NVFP4 (E2M1) 稠密 GEMM，使用 cuBLASLt 块缩放路径（VEC16_UE4M3）。
// A/B = CUDA_R_4F_E2M1（2 元素/字节），C/D = CUDA_R_16BF，compute = FP32。
// TN 布局（A 转置、B 不转置），与 int8/fp8 一致。需要 Blackwell (CC 10.0+)。
// UE4M3 缩放因子填 1.0，不影响数值结果，便于用固定输入值校验输出。
Result run_nvfp4_size(
    cublasLtHandle_t handle, cudaStream_t stream, int size, int iters,
    int warmup, std::size_t requested_workspace) {
  Result result;
  result.precision = "nvfp4";
  result.input_type = "FP4_E2M1";
  result.accumulator_type = "FP32";
  result.output_type = "BF16";
  result.unit = "TFLOPS";
  result.m = result.n = result.k = size;
  result.algorithm = "cuBLASLt NVFP4 block-scaled (VEC16_UE4M3) layout=TN";

  __nv_fp4x2_storage_t* a = nullptr;
  __nv_fp4x2_storage_t* b = nullptr;
  __nv_bfloat16* d = nullptr;
  void* a_scale = nullptr;
  void* b_scale = nullptr;
  void* workspace = nullptr;
  std::size_t workspace_bytes = requested_workspace;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatrixLayout_t d_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;

  auto cleanup = [&]() {
    if (start) cudaEventDestroy(start);
    if (end) cudaEventDestroy(end);
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
    if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (workspace) cudaFree(workspace);
    if (a_scale) cudaFree(a_scale);
    if (b_scale) cudaFree(b_scale);
    if (a) cudaFree(a);
    if (b) cudaFree(b);
    if (d) cudaFree(d);
  };

  // FP4：每 2 个元素 1 字节。方阵 size 为偶数（测试尺寸 4096/8192/16384 均是）。
  const std::size_t element_count =
      static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
  const std::size_t a_bytes = element_count / 2;
  const std::size_t d_bytes = element_count * sizeof(__nv_bfloat16);
  const std::size_t a_scale_bytes = nvfp4_scale_bytes(size);
  const std::size_t b_scale_bytes = nvfp4_scale_bytes(size);

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (cuda_status != cudaSuccess) {
    result.message = "cudaMemGetInfo failed: " + cuda_error(cuda_status);
    cleanup();
    return result;
  }
  const long double required = static_cast<long double>(a_bytes) +
      static_cast<long double>(a_bytes) +  // B 与 A 同尺寸
      static_cast<long double>(d_bytes) +
      static_cast<long double>(a_scale_bytes) +
      static_cast<long double>(b_scale_bytes);
  if (required + requested_workspace >
      static_cast<long double>(free_bytes) * 0.75L) {
    result.status = "SKIPPED";
    result.message = "Matrix and workspace would use more than 75% of free GPU memory";
    cleanup();
    return result;
  }

  if ((cuda_status = cudaMalloc(&a, a_bytes)) != cudaSuccess ||
      (cuda_status = cudaMalloc(&b, a_bytes)) != cudaSuccess ||
      (cuda_status = cudaMalloc(&d, d_bytes)) != cudaSuccess ||
      (cuda_status = cudaMalloc(&a_scale, a_scale_bytes)) != cudaSuccess ||
      (cuda_status = cudaMalloc(&b_scale, b_scale_bytes)) != cudaSuccess) {
    result.message = "NVFP4 allocation failed: " + cuda_error(cuda_status);
    cudaGetLastError();
    cleanup();
    return result;
  }
  if (workspace_bytes > 0) {
    cuda_status = cudaMalloc(&workspace, workspace_bytes);
    if (cuda_status != cudaSuccess) {
      cudaGetLastError();
      workspace = nullptr;
      workspace_bytes = 0;
    }
  }
  result.workspace_bytes = workspace_bytes;

  // 输入值 0.5 在 FP4 E2M1 中可精确表示（E2M1 表示范围 ±{0,0.5,1,1.5,2,3,4,6}）。
  const float input_value = 0.5f;
  const int fill_threads = 256;
  const std::size_t pair_count = element_count / 2;
  const std::size_t fill_blocks =
      (pair_count + fill_threads - 1) / fill_threads;
  fill_fp4_kernel<<<static_cast<int>(fill_blocks), fill_threads, 0, stream>>>(
      a, pair_count, input_value);
  fill_fp4_kernel<<<static_cast<int>(fill_blocks), fill_threads, 0, stream>>>(
      b, pair_count, input_value);
  if ((cuda_status = fill_buffer(
          d, element_count, CUDA_R_16BF, 0.0f, stream)) != cudaSuccess ||
      (cuda_status = cudaStreamSynchronize(stream)) != cudaSuccess) {
    result.message = "NVFP4 buffer initialization failed: " + cuda_error(cuda_status);
    cleanup();
    return result;
  }

  // UE4M3 缩放因子填 1.0：CUDA_R_8F_UE4M3 与 CUDA_R_8F_E4M3 同编码，
  // 1.0 在 E4M3 中可精确表示。用 __nv_cvt_float_to_fp8 生成（默认最近偶舍入）。
  const uint8_t scale_one = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
      1.0f, __NV_SATFINITE, __NV_E4M3));
  std::vector<uint8_t> a_scale_host(a_scale_bytes, scale_one);
  std::vector<uint8_t> b_scale_host(b_scale_bytes, scale_one);
  if ((cuda_status = cudaMemcpy(
          a_scale, a_scale_host.data(), a_scale_bytes,
          cudaMemcpyHostToDevice)) != cudaSuccess ||
      (cuda_status = cudaMemcpy(
          b_scale, b_scale_host.data(), b_scale_bytes,
          cudaMemcpyHostToDevice)) != cudaSuccess) {
    result.message = "NVFP4 scale upload failed: " + cuda_error(cuda_status);
    cleanup();
    return result;
  }

  cublasStatus_t blas_status =
      cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    result.message = "Matmul descriptor creation failed: " + cublas_error(blas_status);
    cleanup();
    return result;
  }
  // TN 布局：A 转置、B 不转置（方阵 ld=size 不变）。
  cublasOperation_t transpose_a = CUBLAS_OP_T;
  cublasOperation_t transpose_b = CUBLAS_OP_N;
  const int32_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  if ((blas_status = cublasLtMatmulDescSetAttribute(
          operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_a,
          sizeof(transpose_a))) != CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatmulDescSetAttribute(
          operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_b,
          sizeof(transpose_b))) != CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatmulDescSetAttribute(
          operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode,
          sizeof(scale_mode))) != CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatmulDescSetAttribute(
          operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode,
          sizeof(scale_mode))) != CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatmulDescSetAttribute(
          operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale,
          sizeof(a_scale))) != CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatmulDescSetAttribute(
          operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale,
          sizeof(b_scale))) != CUBLAS_STATUS_SUCCESS) {
    result.message = "Matmul descriptor configuration failed: " +
                     cublas_error(blas_status);
    cleanup();
    return result;
  }

  // 布局：A/B 为 4F_E2M1（ld 为逻辑元素数 size），C/D 为 BF16。
  if ((blas_status = cublasLtMatrixLayoutCreate(
          &a_layout, CUDA_R_4F_E2M1, size, size, size)) != CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatrixLayoutCreate(
          &b_layout, CUDA_R_4F_E2M1, size, size, size)) != CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatrixLayoutCreate(
          &c_layout, CUDA_R_16BF, size, size, size)) != CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatrixLayoutCreate(
          &d_layout, CUDA_R_16BF, size, size, size)) != CUBLAS_STATUS_SUCCESS) {
    result.message = "Matrix layout creation failed: " + cublas_error(blas_status);
    cleanup();
    return result;
  }
  if ((blas_status = cublasLtMatmulPreferenceCreate(&preference)) !=
          CUBLAS_STATUS_SUCCESS ||
      (blas_status = cublasLtMatmulPreferenceSetAttribute(
          preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_bytes, sizeof(workspace_bytes))) != CUBLAS_STATUS_SUCCESS) {
    result.message = "Preference configuration failed: " + cublas_error(blas_status);
    cleanup();
    return result;
  }

  constexpr int kRequestedAlgorithms = 16;
  std::vector<cublasLtMatmulHeuristicResult_t> heuristics(kRequestedAlgorithms);
  int returned = 0;
  blas_status = cublasLtMatmulAlgoGetHeuristic(
      handle, operation, a_layout, b_layout, c_layout, d_layout, preference,
      kRequestedAlgorithms, heuristics.data(), &returned);
  if (blas_status != CUBLAS_STATUS_SUCCESS || returned == 0) {
    result.status = "UNSUPPORTED";
    result.message = "No cuBLASLt algorithm for NVFP4: " + cublas_error(blas_status);
    cleanup();
    return result;
  }

  float alpha_f = 1.0f;
  float beta_f = 0.0f;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  int best_index = -1;
  float best_trial_ms = std::numeric_limits<float>::infinity();
  const int trial_count = 2;
  for (int index = 0; index < returned; ++index) {
    if (heuristics[index].state != CUBLAS_STATUS_SUCCESS ||
        heuristics[index].workspaceSize > workspace_bytes) {
      continue;
    }
    blas_status = cublasLtMatmul(
        handle, operation, &alpha_f, a, a_layout, b, b_layout, &beta_f,
        d, c_layout, d, d_layout, &heuristics[index].algo, workspace,
        workspace_bytes, stream);
    if (blas_status != CUBLAS_STATUS_SUCCESS ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }
    cudaEventRecord(start, stream);
    bool trial_ok = true;
    for (int trial = 0; trial < trial_count; ++trial) {
      blas_status = cublasLtMatmul(
          handle, operation, &alpha_f, a, a_layout, b, b_layout, &beta_f,
          d, c_layout, d, d_layout, &heuristics[index].algo, workspace,
          workspace_bytes, stream);
      if (blas_status != CUBLAS_STATUS_SUCCESS) {
        trial_ok = false;
        break;
      }
    }
    if (!trial_ok) continue;
    cudaEventRecord(end, stream);
    if (cudaEventSynchronize(end) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, end);
    const float per_iteration = elapsed / trial_count;
    if (per_iteration < best_trial_ms) {
      best_trial_ms = per_iteration;
      best_index = index;
    }
  }
  if (best_index < 0) {
    result.status = "UNSUPPORTED";
    result.message = "All cuBLASLt heuristic algorithms failed for NVFP4";
    cleanup();
    return result;
  }

  const auto& selected = heuristics[best_index];
  result.workspace_bytes = selected.workspaceSize;

  for (int iteration = 0; iteration < warmup; ++iteration) {
    blas_status = cublasLtMatmul(
        handle, operation, &alpha_f, a, a_layout, b, b_layout, &beta_f,
        d, c_layout, d, d_layout, &selected.algo, workspace,
        workspace_bytes, stream);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
      result.message = "Warmup failed: " + cublas_error(blas_status);
      cleanup();
      return result;
    }
  }
  if ((cuda_status = cudaStreamSynchronize(stream)) != cudaSuccess) {
    result.message = "Warmup synchronization failed: " + cuda_error(cuda_status);
    cleanup();
    return result;
  }

  std::vector<float> times;
  times.reserve(iters);
  for (int iteration = 0; iteration < iters; ++iteration) {
    cudaEventRecord(start, stream);
    blas_status = cublasLtMatmul(
        handle, operation, &alpha_f, a, a_layout, b, b_layout, &beta_f,
        d, c_layout, d, d_layout, &selected.algo, workspace,
        workspace_bytes, stream);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
      result.message = "Timed matmul failed: " + cublas_error(blas_status);
      cleanup();
      return result;
    }
    cudaEventRecord(end, stream);
    if ((cuda_status = cudaEventSynchronize(end)) != cudaSuccess) {
      result.message = "Timed synchronization failed: " + cuda_error(cuda_status);
      cleanup();
      return result;
    }
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, end);
    times.push_back(elapsed);
  }

  result.median_ms = percentile(times, 0.5);
  result.p90_ms = percentile(times, 0.9);
  const long double operations =
      2.0L * static_cast<long double>(size) * size * size;
  result.throughput = static_cast<double>(
      operations / (static_cast<long double>(result.median_ms) / 1000.0L) /
      1.0e12L);

  // 校验：scale=1.0，输入值 0.5，expected = size * 0.5 * 0.5。
  const double expected =
      static_cast<double>(size) * input_value * input_value;
  std::string validation_message;
  const bool valid = validate_first_element(
      d, CUDA_R_16BF, expected, 0.03, &validation_message);
  result.validation = valid ? "PASS" : "FAIL";
  result.status = valid ? "OK" : "VALIDATION_FAILED";
  result.message = validation_message;
  cleanup();
  return result;
}
#endif  // GPU_BENCH_HAS_FP4

std::vector<int> sizes_for(const std::string& precision, bool quick) {
  if (precision == "fp64") return quick
      ? std::vector<int>{4096}
      : std::vector<int>{2048, 4096, 6144};
  if (precision == "int16") return quick
      ? std::vector<int>{4096}
      : std::vector<int>{2048, 4096, 8192};
  if (precision == "tf32" || precision == "fp32") return quick
      ? std::vector<int>{8192}
      : std::vector<int>{4096, 8192, 12288};
  return quick
      ? std::vector<int>{8192}
      : std::vector<int>{4096, 8192, 16384};
}

std::vector<PrecisionSpec> precision_specs() {
  std::vector<PrecisionSpec> specs;
  specs.push_back({
      "int8", "INT8", "INT32", "INT32", "TOPS",
      CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I,
      CUBLAS_COMPUTE_32I, CUDA_R_32I, 1.0, 70});
#if GPU_BENCH_HAS_FP8
  specs.push_back({
      "fp8_e4m3", "FP8_E4M3", "FP32", "FP16", "TFLOPS",
      CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16F, CUDA_R_16F,
      CUBLAS_COMPUTE_32F, CUDA_R_32F, 0.5, 89});
  specs.push_back({
      "fp8_e5m2", "FP8_E5M2", "FP32", "FP16", "TFLOPS",
      CUDA_R_8F_E5M2, CUDA_R_8F_E5M2, CUDA_R_16F, CUDA_R_16F,
      CUBLAS_COMPUTE_32F, CUDA_R_32F, 0.5, 89});
#endif
  specs.push_back({
      "fp16", "FP16", "FP32", "FP16", "TFLOPS",
      CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
      CUBLAS_COMPUTE_32F, CUDA_R_32F, 0.5, 70});
  specs.push_back({
      "bf16", "BF16", "FP32", "BF16", "TFLOPS",
      CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
      CUBLAS_COMPUTE_32F, CUDA_R_32F, 0.5, 80});
  specs.push_back({
      "tf32", "FP32", "FP32", "FP32", "TFLOPS",
      CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
      CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, 0.5, 80});
  specs.push_back({
      "fp32", "FP32", "FP32", "FP32", "TFLOPS",
      CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
      CUBLAS_COMPUTE_32F_PEDANTIC, CUDA_R_32F, 0.5, 50});
  specs.push_back({
      "fp64", "FP64", "FP64", "FP64", "TFLOPS",
      CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F,
      CUBLAS_COMPUTE_64F, CUDA_R_64F, 0.5, 60});
  return specs;
}

const PrecisionSpec* find_spec(
    const std::vector<PrecisionSpec>& specs, const std::string& name) {
  for (const auto& spec : specs) {
    if (spec.name == name) return &spec;
  }
  return nullptr;
}

Result unsupported_result(
    const std::string& precision, const std::string& message) {
  Result result;
  result.precision = precision;
  result.status = "UNSUPPORTED";
  result.message = message;
  result.validation = "NOT_RUN";
  return result;
}

void mark_best(std::vector<Result>* results, const std::string& precision) {
  auto best = results->end();
  for (auto iterator = results->begin(); iterator != results->end(); ++iterator) {
    if (iterator->precision != precision || iterator->status != "OK" ||
        iterator->validation != "PASS") {
      continue;
    }
    if (best == results->end() || iterator->throughput > best->throughput) {
      best = iterator;
    }
  }
  if (best != results->end()) best->best = true;
}

bool parse_options(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&]() -> const char* {
      if (index + 1 >= argc) return nullptr;
      return argv[++index];
    };
    if (argument == "--device") {
      const char* value = next();
      if (!value) return false;
      options->device = std::atoi(value);
    } else if (argument == "--gpus") {
      const char* value = next();
      if (!value) return false;
      options->gpus = value;
    } else if (argument == "--mode") {
      const char* value = next();
      if (!value) return false;
      options->mode = value;
    } else if (argument == "--iters") {
      const char* value = next();
      if (!value) return false;
      options->iters = std::atoi(value);
    } else if (argument == "--warmup") {
      const char* value = next();
      if (!value) return false;
      options->warmup = std::atoi(value);
    } else if (argument == "--workspace-mb") {
      const char* value = next();
      if (!value) return false;
      options->workspace_mb = std::atoi(value);
    } else if (argument == "--timeout") {
      const char* value = next();
      if (!value) return false;
      options->timeout = std::atoi(value);
    } else if (argument == "--output") {
      const char* value = next();
      if (!value) return false;
      options->output = value;
    } else if (argument == "--precisions") {
      const char* value = next();
      if (!value) return false;
      options->precisions = value;
    } else if (argument == "--quick") {
      options->quick = true;
    } else if (argument == "--help") {
      std::cout
          << "Usage: gpu_dense_bench [options]\n"
          << "  --device N          only test a single GPU (overrides --gpus)\n"
          << "  --gpus all|0,1,2    GPU list to test (default: all)\n"
          << "  --mode both|isolated|concurrent\n"
          << "  --precisions int4,int8,fp8_e4m3,fp8_e5m2,nvfp4,fp16,bf16,tf32,fp32,fp64\n"
          << "  --iters N --warmup N --workspace-mb N --timeout N\n"
          << "  --quick\n"
          << "  --output DIR        result directory (default: results/<UTC timestamp>)\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << argument << "\n";
      return false;
    }
  }
  return options->iters > 0 && options->warmup >= 0 &&
         options->workspace_mb >= 0;
}

void write_json(
    const std::string& path, const Options& options,
    const cudaDeviceProp& properties, const std::vector<Result>& results) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Cannot open output file: " + path);
  }
  out << "{\n";
  out << "  \"schema_version\": 1,\n";
  out << "  \"generated_utc\": \"" << utc_timestamp() << "\",\n";
  out << "  \"device_id\": " << options.device << ",\n";
  out << "  \"device\": {\n";
  out << "    \"id\": " << options.device << ",\n";
  out << "    \"name\": \"" << json_escape(properties.name) << "\",\n";
  out << "    \"compute_capability\": \"" << properties.major << "."
      << properties.minor << "\",\n";
  out << "    \"total_memory_bytes\": "
      << static_cast<unsigned long long>(properties.totalGlobalMem) << ",\n";
  out << "    \"multiprocessor_count\": " << properties.multiProcessorCount
      << "\n";
  out << "  },\n";
  out << "  \"configuration\": {\n";
  out << "    \"iters\": " << options.iters << ",\n";
  out << "    \"warmup\": " << options.warmup << ",\n";
  out << "    \"workspace_mb\": " << options.workspace_mb << ",\n";
  out << "    \"quick\": " << (options.quick ? "true" : "false") << ",\n";
  out << "    \"sparsity\": \"none\"\n";
  out << "  },\n";
  out << "  \"results\": [\n";
  for (std::size_t index = 0; index < results.size(); ++index) {
    const auto& result = results[index];
    out << "    {\n";
    out << "      \"precision\": \"" << json_escape(result.precision) << "\",\n";
    out << "      \"input_type\": \"" << json_escape(result.input_type) << "\",\n";
    out << "      \"accumulator_type\": \""
        << json_escape(result.accumulator_type) << "\",\n";
    out << "      \"output_type\": \"" << json_escape(result.output_type) << "\",\n";
    out << "      \"m\": " << result.m << ", \"n\": " << result.n
        << ", \"k\": " << result.k << ",\n";
    out << std::fixed << std::setprecision(6);
    out << "      \"median_ms\": " << result.median_ms << ",\n";
    out << "      \"p90_ms\": " << result.p90_ms << ",\n";
    out << "      \"throughput\": " << result.throughput << ",\n";
    out << "      \"unit\": \"" << json_escape(result.unit) << "\",\n";
    out << "      \"best\": " << (result.best ? "true" : "false") << ",\n";
    out << "      \"validation\": \"" << json_escape(result.validation) << "\",\n";
    out << "      \"status\": \"" << json_escape(result.status) << "\",\n";
    out << "      \"algorithm\": \"" << json_escape(result.algorithm) << "\",\n";
    out << "      \"workspace_bytes\": "
        << static_cast<unsigned long long>(result.workspace_bytes) << ",\n";
    out << "      \"message\": \"" << json_escape(result.message) << "\"\n";
    out << "    }" << (index + 1 < results.size() ? "," : "") << "\n";
  }
  out << "  ]\n";
  out << "}\n";
}

// ---- 多卡顶层报告写入器（对照 run_multi_gpu.py 1:1 复刻）----

// 精度显示排序键：已知精度按 PRECISION_ORDER 索引，未知排末尾按字母序。
// 对应 Python precision_sort_key()。返回 (group, index_or_name) 供比较。
struct PrecisionSortKey {
  int group;                 // 0 = 已知，1 = 未知
  int index;                 // group==0 时的索引
  std::string name;          // group==1 时的名称
  bool operator<(const PrecisionSortKey& other) const {
    if (group != other.group) return group < other.group;
    if (group == 0) return index < other.index;
    return name < other.name;
  }
};
PrecisionSortKey precision_sort_key(const std::string& name) {
  static const std::vector<std::string> order = {
      "fp64", "fp32", "tf32", "bf16", "fp16",
      "int8", "fp8_e4m3", "fp8_e5m2", "nvfp4", "int4"};
  for (std::size_t i = 0; i < order.size(); ++i) {
    if (order[i] == name) return {0, static_cast<int>(i), std::string()};
  }
  return {1, 0, name};
}

// 遥测 map 序列化为 JSON 对象（键值均为字符串）。
void write_telemetry_json(std::ofstream& out, const std::string& indent,
                          const std::map<std::string, std::string>& telemetry) {
  out << "{";
  bool first = true;
  for (const auto& kv : telemetry) {
    if (!first) out << ", ";
    first = false;
    out << "\"" << json_escape(kv.first) << "\": \"" << json_escape(kv.second)
        << "\"";
  }
  out << "}";
}

// 顶层 JSON 报告。对应 Python main() 的 payload 结构。
void write_report_json(const std::string& path, const Environment& env,
                       const std::vector<int>& selected_gpus,
                       const Options& options,
                       const std::vector<DeviceReport>& runs) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot open output file: " + path);
  out << "{\n";
  out << "  \"schema_version\": 1,\n";
  out << "  \"generated_utc\": \"" << iso_timestamp_ms() << "\",\n";
  // environment
  out << "  \"environment\": {\n";
  out << "    \"driver_version\": \"" << json_escape(env.driver_version)
      << "\",\n";
  out << "    \"nvcc_version\": \"" << json_escape(env.nvcc_version) << "\"\n";
  out << "  },\n";
  // selected_gpus
  out << "  \"selected_gpus\": [";
  for (std::size_t i = 0; i < selected_gpus.size(); ++i) {
    if (i) out << ", ";
    out << selected_gpus[i];
  }
  out << "],\n";
  // gpu_inventory
  out << "  \"gpu_inventory\": [\n";
  for (std::size_t i = 0; i < env.gpu_inventory.size(); ++i) {
    const auto& g = env.gpu_inventory[i];
    out << "    {\n";
    out << "      \"index\": " << g.index << ",\n";
    out << "      \"name\": \"" << json_escape(g.name) << "\",\n";
    out << "      \"uuid\": \"" << json_escape(g.uuid) << "\",\n";
    out << "      \"memory.total\": \"" << json_escape(g.memory_total)
        << "\",\n";
    out << "      \"compute_cap\": \"" << json_escape(g.compute_cap) << "\",\n";
    out << "      \"power.limit\": \"" << json_escape(g.power_limit) << "\"\n";
    out << "    }" << (i + 1 < env.gpu_inventory.size() ? "," : "") << "\n";
  }
  out << "  ],\n";
  // configuration
  out << "  \"configuration\": {\n";
  out << "    \"mode\": \"" << options.mode << "\",\n";
  out << "    \"quick\": " << (options.quick ? "true" : "false") << ",\n";
  out << "    \"precisions\": [";
  {
    std::vector<std::string> precs = split(options.precisions, ',');
    for (std::size_t i = 0; i < precs.size(); ++i) {
      if (i) out << ", ";
      out << "\"" << json_escape(precs[i]) << "\"";
    }
  }
  out << "],\n";
  out << "    \"iters\": " << options.iters << ",\n";
  out << "    \"warmup\": " << options.warmup << ",\n";
  out << "    \"workspace_mb\": " << options.workspace_mb << ",\n";
  out << "    \"timeout_seconds\": " << options.timeout << ",\n";
  out << "    \"sparsity\": \"none\"\n";
  out << "  },\n";
  // runs
  out << "  \"runs\": [\n";
  for (std::size_t i = 0; i < runs.size(); ++i) {
    const auto& run = runs[i];
    out << "    {\n";
    out << "      \"mode\": \"" << run.mode << "\",\n";
    out << "      \"device_id\": " << run.device_id << ",\n";
    out << "      \"status\": \"" << json_escape(run.status) << "\",\n";
    if (run.status != "OK") {
      out << "      \"error\": \"" << json_escape(run.error) << "\"\n";
    } else {
      out << "      \"device\": {\n";
      out << "        \"id\": " << run.device_index << ",\n";
      out << "        \"name\": \"" << json_escape(run.device_name) << "\",\n";
      out << "        \"compute_capability\": \""
          << json_escape(run.compute_capability) << "\",\n";
      out << "        \"total_memory_bytes\": " << run.total_memory_bytes
          << ",\n";
      out << "        \"multiprocessor_count\": " << run.multiprocessor_count
          << "\n";
      out << "      },\n";
      out << "      \"results\": [\n";
      const auto& results = run.results;
      for (std::size_t j = 0; j < results.size(); ++j) {
        const auto& r = results[j];
        out << "        {\n";
        out << "          \"precision\": \"" << json_escape(r.precision)
            << "\",\n";
        out << "          \"input_type\": \"" << json_escape(r.input_type)
            << "\",\n";
        out << "          \"accumulator_type\": \""
            << json_escape(r.accumulator_type) << "\",\n";
        out << "          \"output_type\": \"" << json_escape(r.output_type)
            << "\",\n";
        out << "          \"m\": " << r.m << ", \"n\": " << r.n
            << ", \"k\": " << r.k << ",\n";
        out << std::fixed << std::setprecision(6);
        out << "          \"median_ms\": " << r.median_ms << ",\n";
        out << "          \"p90_ms\": " << r.p90_ms << ",\n";
        out << "          \"throughput\": " << r.throughput << ",\n";
        out << "          \"unit\": \"" << json_escape(r.unit) << "\",\n";
        out << "          \"best\": " << (r.best ? "true" : "false") << ",\n";
        out << "          \"validation\": \"" << json_escape(r.validation)
            << "\",\n";
        out << "          \"status\": \"" << json_escape(r.status) << "\",\n";
        out << "          \"algorithm\": \"" << json_escape(r.algorithm)
            << "\",\n";
        out << "          \"workspace_bytes\": "
            << static_cast<unsigned long long>(r.workspace_bytes) << ",\n";
        out << "          \"message\": \"" << json_escape(r.message) << "\"\n";
        out << "        }" << (j + 1 < results.size() ? "," : "") << "\n";
      }
      out << "      ],\n";
      out << "      \"telemetry_before\": ";
      write_telemetry_json(out, "      ", run.telemetry_before);
      out << ",\n";
      out << "      \"telemetry_after\": ";
      write_telemetry_json(out, "      ", run.telemetry_after);
      out << "\n";
    }
    out << "    }" << (i + 1 < runs.size() ? "," : "") << "\n";
  }
  out << "  ]\n";
  out << "}\n";
}

// CSV 报告行：从 DeviceReport 列表提取，对应 Python best_rows()。
// 非 OK 的 run 产生一行存根（gpu_name/precision 为空）；OK 的 run 每个 result 一行。
struct CsvRow {
  std::string mode;
  int device_id;
  std::string gpu_name;
  std::string compute_capability;
  std::string precision;
  std::string input_type;
  std::string accumulator_type;
  std::string output_type;
  int m = 0, n = 0, k = 0;
  double median_ms = 0.0, p90_ms = 0.0, throughput = 0.0;
  std::string unit;
  bool best = false;
  std::string validation;
  std::string status;
  std::string algorithm;
  std::size_t workspace_bytes = 0;
  std::string message;
};

std::vector<CsvRow> build_rows(const std::vector<DeviceReport>& runs) {
  std::vector<CsvRow> rows;
  for (const auto& run : runs) {
    if (run.status != "OK") {
      CsvRow row;
      row.mode = run.mode;
      row.device_id = run.device_id;
      row.gpu_name = "";
      row.precision = "";
      row.status = run.status.empty() ? "ERROR" : run.status;
      row.message = run.error;
      rows.push_back(std::move(row));
      continue;
    }
    for (const auto& r : run.results) {
      CsvRow row;
      row.mode = run.mode;
      row.device_id = run.has_device ? run.device_index : run.device_id;
      row.gpu_name = run.has_device ? run.device_name : "unknown";
      row.compute_capability = run.has_device ? run.compute_capability : "unknown";
      row.precision = r.precision;
      row.input_type = r.input_type;
      row.accumulator_type = r.accumulator_type;
      row.output_type = r.output_type;
      row.m = r.m;
      row.n = r.n;
      row.k = r.k;
      row.median_ms = r.median_ms;
      row.p90_ms = r.p90_ms;
      row.throughput = r.throughput;
      row.unit = r.unit;
      row.best = r.best;
      row.validation = r.validation;
      row.status = r.status;
      row.algorithm = r.algorithm;
      row.workspace_bytes = r.workspace_bytes;
      row.message = r.message;
      rows.push_back(std::move(row));
    }
  }
  return rows;
}

// CSV 字段转义：含逗号/引号/换行时用双引号包裹，内部双引号翻倍。
std::string csv_field(const std::string& value) {
  bool need_quote = value.find(',') != std::string::npos ||
                    value.find('"') != std::string::npos ||
                    value.find('\n') != std::string::npos ||
                    value.find('\r') != std::string::npos;
  if (!need_quote) return value;
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '"') out += "\"\"";
    else out += ch;
  }
  out += "\"";
  return out;
}

// CSV 报告。21 列，顺序与 Python write_csv() 完全一致。
void write_report_csv(const std::string& path,
                      const std::vector<DeviceReport>& runs) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot open output file: " + path);
  out << "mode,device_id,gpu_name,compute_capability,precision,input_type,"
      << "accumulator_type,output_type,m,n,k,median_ms,p90_ms,throughput,"
      << "unit,best,validation,status,algorithm,workspace_bytes,message\n";
  out << std::fixed << std::setprecision(6);
  for (const auto& row : build_rows(runs)) {
    out << csv_field(row.mode) << "," << row.device_id << ","
        << csv_field(row.gpu_name) << ","
        << csv_field(row.compute_capability) << ","
        << csv_field(row.precision) << "," << csv_field(row.input_type) << ","
        << csv_field(row.accumulator_type) << ","
        << csv_field(row.output_type) << "," << row.m << "," << row.n << ","
        << row.k << "," << row.median_ms << "," << row.p90_ms << ","
        << row.throughput << "," << csv_field(row.unit) << ","
        << (row.best ? "True" : "False") << "," << csv_field(row.validation)
        << "," << csv_field(row.status) << "," << csv_field(row.algorithm)
        << "," << static_cast<unsigned long long>(row.workspace_bytes) << ","
        << csv_field(row.message) << "\n";
  }
}

// 数值格式化为定小数位，非数值返回空（对应 Python format_value）。
std::string format_value(double value, int digits) {
  std::ostringstream s;
  s << std::fixed << std::setprecision(digits) << value;
  return s.str();
}

// 左对齐填充/右对齐填充（对应 Python {:<N} / {:>N}）。
std::string pad_left(const std::string& s, std::size_t width) {
  return (s.size() >= width) ? s : s + std::string(width - s.size(), ' ');
}
std::string pad_right(const std::string& s, std::size_t width) {
  return (s.size() >= width) ? s : std::string(width - s.size(), ' ') + s;
}

// Markdown 报告。1:1 复刻 Python write_text()。
void write_report_md(const std::string& path, const Environment& env,
                     const std::vector<int>& /*selected_gpus*/,
                     const Options& options,
                     const std::vector<DeviceReport>& runs) {
  std::vector<CsvRow> rows = build_rows(runs);
  std::ostringstream out;

  // 头部信息块。
  out << "Dense GPU GEMM Benchmark\n";
  out << "Generated: " << iso_timestamp_ms() << "\n";
  out << "Driver: " << env.driver_version << "\n";
  out << "CUDA compiler: " << env.nvcc_version << "\n";
  out << "Dense only: no 2:4 sparsity or sparse GEMM APIs\n";
  out << "\n";

  // best 行与 unsupported 行。
  std::vector<const CsvRow*> best;
  std::vector<const CsvRow*> unsupported;
  for (const auto& row : rows) {
    if (row.best) {
      best.push_back(&row);
    } else if (row.status == "UNSUPPORTED" || row.status == "SKIPPED") {
      unsupported.push_back(&row);
    }
  }

  // [ISOLATED] / [CONCURRENT] 分节。
  for (const std::string& mode : {"isolated", "concurrent"}) {
    std::vector<const CsvRow*> mode_rows;
    for (const auto* r : best) {
      if (r->mode == mode) mode_rows.push_back(r);
    }
    if (mode_rows.empty()) continue;

    out << "[" << mode << "]\n";
    // 该模式下的所有 device_id（升序）。
    std::set<int> device_ids;
    for (const auto* r : mode_rows) device_ids.insert(r->device_id);
    for (int device : device_ids) {
      std::vector<const CsvRow*> gpu_rows;
      for (const auto* r : mode_rows) {
        if (r->device_id == device) gpu_rows.push_back(r);
      }
      std::string name = gpu_rows.empty() ? "unknown" : gpu_rows[0]->gpu_name;
      if (name.empty()) name = "unknown";
      out << "GPU " << device << ": " << name << "\n";
      out << "  Precision      M=N=K      Median(ms)      Throughput       Validation\n";
      // 按精度排序。
      std::sort(gpu_rows.begin(), gpu_rows.end(),
                [](const CsvRow* a, const CsvRow* b) {
                  return precision_sort_key(a->precision) <
                         precision_sort_key(b->precision);
                });
      for (const auto* r : gpu_rows) {
        std::string precision = r->precision;
        std::string size_str = std::to_string(r->m);
        std::string median = format_value(r->median_ms, 3);
        std::string throughput = format_value(r->throughput, 2);
        std::string unit = r->unit;
        std::string validation = r->validation;
        out << "  " << pad_left(precision, 14) << " " << pad_left(size_str, 10)
            << " " << pad_left(median, 15) << " " << pad_right(throughput, 10)
            << " " << pad_left(unit, 8) << " " << validation << "\n";
      }
      out << "\n";
    }

    // concurrent aggregate（仅 concurrent 模式）。
    if (mode == "concurrent") {
      out << "  Concurrent aggregate:\n";
      // 收集该模式下所有精度（去重），按精度排序。
      std::set<std::string> precisions;
      for (const auto* r : mode_rows) precisions.insert(r->precision);
      std::vector<std::string> sorted_precs(precisions.begin(), precisions.end());
      std::sort(sorted_precs.begin(), sorted_precs.end(),
                [](const std::string& a, const std::string& b) {
                  return precision_sort_key(a) < precision_sort_key(b);
                });
      for (const std::string& prec : sorted_precs) {
        std::vector<const CsvRow*> prec_rows;
        for (const auto* r : mode_rows) {
          if (r->precision == prec) prec_rows.push_back(r);
        }
        // 单位必须一致才输出聚合行。
        std::set<std::string> units;
        for (const auto* r : prec_rows) units.insert(r->unit);
        if (units.size() != 1) continue;
        double total = 0.0;
        for (const auto* r : prec_rows) total += r->throughput;
        std::ostringstream total_s;
        total_s << std::fixed << std::setprecision(2) << total;
        out << "    " << pad_left(prec, 14) << " " << pad_right(total_s.str(), 10)
            << " " << *units.begin() << "\n";
      }
      out << "\n";
    }
  }

  // [UNSUPPORTED / SKIPPED] 分节（按 mode/device/precision/message 去重）。
  if (!unsupported.empty()) {
    out << "[UNSUPPORTED / SKIPPED]\n";
    std::set<std::tuple<std::string, int, std::string, std::string>> seen;
    for (const auto* r : unsupported) {
      auto key = std::make_tuple(r->mode, r->device_id, r->precision, r->message);
      if (seen.count(key)) continue;
      seen.insert(key);
      out << "  " << r->mode << " GPU " << r->device_id << " " << r->precision
          << ": " << r->status << " - " << r->message << "\n";
    }
    out << "\n";
  }

  // [ERRORS] 分节。
  std::vector<const CsvRow*> errors;
  for (const auto& row : rows) {
    if (row.status == "ERROR" || row.status == "TIMEOUT" ||
        row.status == "VALIDATION_FAILED") {
      errors.push_back(&row);
    }
  }
  if (!errors.empty()) {
    out << "[ERRORS]\n";
    for (const auto* r : errors) {
      out << "  " << r->mode << " GPU " << r->device_id << ": " << r->status
          << " - " << r->message << "\n";
    }
  }

  std::ofstream file(path);
  if (!file) throw std::runtime_error("Cannot open output file: " + path);
  file << out.str();
}

// 在单张 GPU 上跑全部精度，返回完整 DeviceReport。自给自足：每线程各自
// cudaSetDevice + 自建 handle/stream + 跑循环 + 销毁，无共享状态，线程安全。
// 对应 Python run_one() 的成功路径（status=OK 时填充 device/results/telemetry）。
DeviceReport run_device(int device_id, const Options& options) {
  DeviceReport report;
  report.device_id = device_id;
  report.has_device = false;

  cudaError_t cuda_status = cudaSetDevice(device_id);
  if (cuda_status != cudaSuccess) {
    report.status = "ERROR";
    report.error = "cudaSetDevice failed: " + cuda_error(cuda_status);
    return report;
  }
  cudaDeviceProp properties{};
  if ((cuda_status = cudaGetDeviceProperties(&properties, device_id)) != cudaSuccess) {
    report.status = "ERROR";
    report.error = "cudaGetDeviceProperties failed: " + cuda_error(cuda_status);
    return report;
  }
  report.has_device = true;
  report.device_index = device_id;
  report.device_name = properties.name;
  {
    std::ostringstream cc;
    cc << properties.major << "." << properties.minor;
    report.compute_capability = cc.str();
  }
  report.total_memory_bytes = static_cast<unsigned long long>(properties.totalGlobalMem);
  report.multiprocessor_count = properties.multiProcessorCount;

  cublasLtHandle_t handle = nullptr;
  cublasStatus_t blas_status = cublasLtCreate(&handle);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    report.status = "ERROR";
    report.error = "cublasLtCreate failed: " + cublas_error(blas_status);
    return report;
  }
  cudaStream_t stream = nullptr;
  if ((cuda_status = cudaStreamCreate(&stream)) != cudaSuccess) {
    report.status = "ERROR";
    report.error = "cudaStreamCreate failed: " + cuda_error(cuda_status);
    cublasLtDestroy(handle);
    return report;
  }

  // 跑之前采一次遥测（与 Python run_one 的 telemetry_before 一致）。
  report.telemetry_before = read_telemetry(device_id);

  std::vector<Result>& results = report.results;
  const auto specs = precision_specs();
  const int compute_capability = properties.major * 10 + properties.minor;
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(options.workspace_mb) * 1024ULL * 1024ULL;

  for (const auto& requested : split(options.precisions, ',')) {
    if (requested == "int16") {
      for (int size : sizes_for(requested, options.quick)) {
        std::cout << "GPU " << device_id << " INT16 " << size << "^3\n";
        results.push_back(run_int16_size(
            stream, size, options.iters, options.warmup));
      }
      mark_best(&results, requested);
      continue;
    }
    if (requested == "nvfp4") {
#if GPU_BENCH_HAS_FP4
      // NVFP4 需要 Blackwell (CC 10.0+)。
      if (compute_capability < 100) {
        std::ostringstream message;
        message << "Requires compute capability 10.0 or newer (Blackwell); "
                   "device is "
                << properties.major << "." << properties.minor;
        results.push_back(unsupported_result(requested, message.str()));
      } else {
        for (int size : sizes_for(requested, options.quick)) {
          std::cout << "GPU " << device_id << " NVFP4 " << size << "^3\n";
          results.push_back(run_nvfp4_size(
              handle, stream, size, options.iters, options.warmup,
              workspace_bytes));
        }
        mark_best(&results, requested);
      }
#else
      results.push_back(unsupported_result(
          requested,
          "NVFP4 requires CUDA Toolkit 13.0 or newer at build time "
          "and Blackwell (CC 10.0+) hardware"));
#endif
      continue;
    }
    if (requested == "int4") {
      // cuBLASLt 不提供 INT4 稠密 GEMM 内核（CUDA_R_4I 在各 compute/output 组合
      // 与布局下均返回 CUBLAS_STATUS_NOT_SUPPORTED，已在多架构上实测确认）。
      // 5090 等 Blackwell 硬件虽支持 INT4 Tensor Core，但仅通过推理框架
      // （CUTLASS / TensorRT-LLM 等）以分组量化路径使用，而非标准稠密 GEMM API。
      // 量化推理（如 AWQ-INT4）的实际计算是 INT4 权重反量化回 FP16/BF16 再做
      // GEMM，并非纯 INT4 Tensor Core 稠密乘加，故此处不伪造数值。
      results.push_back(unsupported_result(
          requested,
          "cuBLASLt does not provide a dense INT4 GEMM kernel; INT4 Tensor "
          "Core is used via inference frameworks (CUTLASS/quantized paths), "
          "not standard dense GEMM APIs"));
      continue;
    }
    const PrecisionSpec* spec = find_spec(specs, requested);
    if (!spec) {
#if !GPU_BENCH_HAS_FP8
      if (requested == "fp8_e4m3" || requested == "fp8_e5m2") {
        results.push_back(unsupported_result(
            requested,
            "FP8 requires CUDA Toolkit 13.0 or newer at build time"));
        continue;
      }
#endif
      results.push_back(unsupported_result(
          requested, "Unknown precision name or unavailable in this CUDA build"));
      continue;
    }
    if (compute_capability < spec->min_cc) {
      std::ostringstream message;
      message << "Requires compute capability " << spec->min_cc / 10 << "."
              << spec->min_cc % 10 << " or newer; device is "
              << properties.major << "." << properties.minor;
      results.push_back(unsupported_result(requested, message.str()));
      continue;
    }
    for (int size : sizes_for(requested, options.quick)) {
      std::cout << "GPU " << device_id << " " << requested << " "
                << size << "^3\n";
      results.push_back(run_cublaslt_size(
          handle, stream, *spec, size, options.iters, options.warmup,
          workspace_bytes));
    }
    mark_best(&results, requested);
  }

  cudaStreamDestroy(stream);
  cublasLtDestroy(handle);

  // 跑之后采一次遥测（仅成功路径才填充，与 Python 一致）。
  report.telemetry_after = read_telemetry(device_id);
  return report;
}

// 解析 GPU 选择：--device N 优先；否则 --gpus all|0,1,2。校验卡号有效。
std::vector<int> select_gpus(const Options& options, int device_count) {
  std::vector<int> selected;
  if (options.device >= 0) {
    if (options.device >= device_count) return selected;  // 无效，调用方报错
    selected.push_back(options.device);
    return selected;
  }
  if (options.gpus == "all") {
    for (int i = 0; i < device_count; ++i) selected.push_back(i);
    return selected;
  }
  for (const std::string& token : split(options.gpus, ',')) {
    if (token.empty()) continue;
    try {
      selected.push_back(std::stoi(token));
    } catch (...) {
      return {};  // 解析失败
    }
  }
  // 去重 + 排序 + 校验范围。
  std::set<int> unique_set(selected.begin(), selected.end());
  selected.assign(unique_set.begin(), unique_set.end());
  for (int id : selected) {
    if (id < 0 || id >= device_count) return {};
  }
  return selected;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, &options)) {
    std::cerr << "Invalid arguments. Run with --help.\n";
    return 2;
  }
  if (options.mode != "both" && options.mode != "isolated" &&
      options.mode != "concurrent") {
    std::cerr << "Invalid --mode: " << options.mode
              << " (expected both/isolated/concurrent)\n";
    return 2;
  }

  int device_count = 0;
  cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cuda_error(cuda_status) << "\n";
    return 3;
  }
  std::vector<int> selected = select_gpus(options, device_count);
  if (selected.empty()) {
    std::cerr << "No valid GPU selected. --device/--gpus must reference an "
                 "existing GPU (device count is "
              << device_count << ")\n";
    return 4;
  }

  // 输出目录：默认 results/<UTC 时间戳>（对齐 Python %Y%m%dT%H%M%SZ 紧凑格式）。
  std::string output_dir = options.output;
  if (output_dir.empty()) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t stamp = std::chrono::system_clock::to_time_t(now);
    std::tm tm_value{};
#ifdef _WIN32
    gmtime_s(&tm_value, &stamp);
#else
    gmtime_r(&stamp, &tm_value);
#endif
    std::ostringstream dir;
    dir << "results/" << std::put_time(&tm_value, "%Y%m%dT%H%M%SZ");
    output_dir = dir.str();
  }
  // 创建输出目录（递归）。跨平台简单实现。
  std::string mkdir_cmd =
#ifdef _WIN32
      "if not exist \"" + output_dir + "\" mkdir \"" + output_dir + "\"";
#else
      "mkdir -p \"" + output_dir + "\"";
#endif
  std::system(mkdir_cmd.c_str());

  std::cout << "Building/running dense GPU GEMM benchmark\n";
  std::cout << "Selected GPUs: ";
  for (std::size_t i = 0; i < selected.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << selected[i];
  }
  std::cout << "  mode: " << options.mode << "\n\n";

  std::vector<DeviceReport> runs;

  // isolated 模式：逐卡串行。
  if (options.mode == "both" || options.mode == "isolated") {
    std::cout << "Running isolated per-GPU measurements...\n";
    for (int device : selected) {
      std::cout << "  GPU " << device << "\n";
      DeviceReport report = run_device(device, options);
      report.mode = "isolated";
      runs.push_back(std::move(report));
    }
  }

  // concurrent 模式：每卡一线程同时跑。每线程返回独立 DeviceReport，主线程汇总，
  // 无共享状态、无锁。对应 Python ThreadPoolExecutor。
  if (options.mode == "both" || options.mode == "concurrent") {
    std::cout << "Running concurrent all-GPU measurements...\n";
    std::vector<DeviceReport> concurrent_reports(selected.size());
    std::vector<std::thread> threads;
    std::vector<std::string> errors(selected.size());  // 防御性隔离异常
    for (std::size_t i = 0; i < selected.size(); ++i) {
      threads.emplace_back([&options, &selected, &concurrent_reports, &errors, i]() {
        try {
          DeviceReport report = run_device(selected[i], options);
          report.mode = "concurrent";
          concurrent_reports[i] = std::move(report);
        } catch (const std::exception& exc) {
          DeviceReport report;
          report.mode = "concurrent";
          report.device_id = selected[i];
          report.status = "ERROR";
          report.error = "Runner exception: " + std::string(exc.what());
          concurrent_reports[i] = std::move(report);
        } catch (...) {
          DeviceReport report;
          report.mode = "concurrent";
          report.device_id = selected[i];
          report.status = "ERROR";
          report.error = "Runner exception: unknown";
          concurrent_reports[i] = std::move(report);
        }
      });
    }
    for (auto& t : threads) t.join();
    for (auto& report : concurrent_reports) {
      std::cout << "  GPU " << report.device_id << " completed\n";
      runs.push_back(std::move(report));
    }
  }

  // 采集环境信息（driver/nvcc/gpu_inventory）。
  Environment env = query_environment();

  // 写三份报告。
  try {
    std::string base = output_dir;
    if (!base.empty() && base.back() != '/' && base.back() != '\\') base += "/";
    write_report_json(base + "report.json", env, selected, options, runs);
    write_report_csv(base + "report.csv", runs);
    write_report_md(base + "report.md", env, selected, options, runs);
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 9;
  }

  std::cout << "Reports written to: " << output_dir << "\n";
  std::cout << "  report.md\n  report.csv\n  report.json\n";

  // 退出码：有 ERROR/TIMEOUT/VALIDATION_FAILED 返回 1，否则 0（与 Python 一致）。
  bool failed = false;
  for (const auto& run : runs) {
    if (run.status == "ERROR" || run.status == "TIMEOUT") {
      failed = true;
      break;
    }
    for (const auto& r : run.results) {
      if (r.status == "ERROR" || r.status == "TIMEOUT" ||
          r.status == "VALIDATION_FAILED") {
        failed = true;
        break;
      }
    }
    if (failed) break;
  }
  return failed ? 1 : 0;
}
