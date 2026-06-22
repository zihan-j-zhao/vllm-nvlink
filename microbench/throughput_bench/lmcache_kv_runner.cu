#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t _err = (expr);                                               \
    if (_err != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(_err));                                \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

#define CU_CHECK(expr)                                                       \
  do {                                                                       \
    CUresult _err = (expr);                                                   \
    if (_err != CUDA_SUCCESS) {                                               \
      const char* _name = nullptr;                                            \
      const char* _str = nullptr;                                             \
      cuGetErrorName(_err, &_name);                                           \
      cuGetErrorString(_err, &_str);                                          \
      std::fprintf(stderr, "CUDA driver error %s:%d: %s (%s)\n", __FILE__,   \
                   __LINE__, _name ? _name : "?", _str ? _str : "?");       \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

struct Args {
  int src_device = -1;
  int dst_device = -1;
  size_t chunk_bytes = 0;
  size_t buffer_bytes = 0;
  int chunks_per_burst = 1;
  long long burst_interval_us = 1000;
  int max_inflight = 64;
  std::string ready_file;
  std::string go_file;
  std::string stop_file;
  std::string step_signal_dir;
  std::string stats_out;
};

bool exists(const std::string& path) {
  if (path.empty()) return false;
  std::ifstream f(path);
  return f.good();
}

void touch(const std::string& path) {
  if (path.empty()) return;
  std::ofstream f(path);
}

long long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

size_t parse_ull(const char* value) {
  return static_cast<size_t>(std::strtoull(value, nullptr, 10));
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", argv[i]);
        std::exit(1);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--src-device") == 0) {
      args.src_device = std::atoi(next());
    } else if (std::strcmp(argv[i], "--dst-device") == 0) {
      args.dst_device = std::atoi(next());
    } else if (std::strcmp(argv[i], "--chunk-bytes") == 0) {
      args.chunk_bytes = parse_ull(next());
    } else if (std::strcmp(argv[i], "--buffer-bytes") == 0) {
      args.buffer_bytes = parse_ull(next());
    } else if (std::strcmp(argv[i], "--chunks-per-burst") == 0) {
      args.chunks_per_burst = std::atoi(next());
    } else if (std::strcmp(argv[i], "--burst-interval-us") == 0) {
      args.burst_interval_us = std::atoll(next());
    } else if (std::strcmp(argv[i], "--max-inflight-copies") == 0) {
      args.max_inflight = std::atoi(next());
    } else if (std::strcmp(argv[i], "--ready-file") == 0) {
      args.ready_file = next();
    } else if (std::strcmp(argv[i], "--go-file") == 0) {
      args.go_file = next();
    } else if (std::strcmp(argv[i], "--stop-file") == 0) {
      args.stop_file = next();
    } else if (std::strcmp(argv[i], "--step-signal-dir") == 0) {
      args.step_signal_dir = next();
    } else if (std::strcmp(argv[i], "--stats-out") == 0) {
      args.stats_out = next();
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      std::exit(1);
    }
  }
  if (args.src_device < 0 || args.dst_device < 0 || args.chunk_bytes == 0 ||
      args.buffer_bytes == 0 || args.chunks_per_burst < 1 ||
      args.burst_interval_us < 0 || args.max_inflight < 1) {
    std::fprintf(stderr, "invalid arguments\n");
    std::exit(1);
  }
  if (args.chunk_bytes > args.buffer_bytes) {
    args.buffer_bytes = args.chunk_bytes;
  }
  return args;
}

void enable_peer_if_possible(int device, int peer) {
  CUDA_CHECK(cudaSetDevice(device));
  int can_access = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device, peer));
  if (!can_access) {
    std::fprintf(stderr, "GPU %d cannot access peer GPU %d\n", device, peer);
    std::exit(2);
  }
  cudaError_t err = cudaDeviceEnablePeerAccess(peer, 0);
  if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
    std::fprintf(stderr, "cudaDeviceEnablePeerAccess(%d -> %d): %s\n", device,
                 peer, cudaGetErrorString(err));
    std::exit(2);
  }
  cudaGetLastError();
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  CU_CHECK(cuInit(0));

  enable_peer_if_possible(args.src_device, args.dst_device);
  enable_peer_if_possible(args.dst_device, args.src_device);

  void* src = nullptr;
  void* dst = nullptr;
  CUDA_CHECK(cudaSetDevice(args.src_device));
  CUDA_CHECK(cudaMalloc(&src, args.buffer_bytes));
  CUDA_CHECK(cudaMemset(src, 0x5a, args.buffer_bytes));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  CUDA_CHECK(cudaSetDevice(args.dst_device));
  CUDA_CHECK(cudaMalloc(&dst, args.buffer_bytes));
  CUDA_CHECK(cudaMemset(dst, 0, args.buffer_bytes));

  CUDA_CHECK(cudaSetDevice(args.src_device));
  std::vector<cudaEvent_t> events(args.max_inflight);
  for (auto& event : events) {
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }
  CUstream driver_stream = reinterpret_cast<CUstream>(stream);

  std::printf("[lmcache_kv_runner] src=cuda:%d dst=cuda:%d chunk=%zu "
              "buffer=%zu chunks_per_burst=%d interval_us=%lld "
              "max_inflight=%d\n",
              args.src_device, args.dst_device, args.chunk_bytes,
              args.buffer_bytes, args.chunks_per_burst,
              args.burst_interval_us, args.max_inflight);
  std::fflush(stdout);

  touch(args.ready_file);
  while (!exists(args.go_file)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  const long long interval_ns = args.burst_interval_us * 1000ll;
  long long next_burst_ns = now_ns();
  const long long start_ns = now_ns();
  long long stop_ns = start_ns;
  unsigned long long bursts = 0;
  unsigned long long chunks = 0;
  unsigned long long bytes = 0;

  auto issue_burst = [&]() {
    ++bursts;
    for (int i = 0; i < args.chunks_per_burst; ++i) {
      cudaEvent_t event = events[chunks % events.size()];
      if (chunks >= events.size()) {
        CUDA_CHECK(cudaEventSynchronize(event));
      }
      CU_CHECK(cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(dst),
                                 reinterpret_cast<CUdeviceptr>(src),
                                 args.chunk_bytes, driver_stream));
      CUDA_CHECK(cudaEventRecord(event, stream));
      ++chunks;
      bytes += args.chunk_bytes;
    }
  };

  unsigned long long step = 0;
  while (!exists(args.stop_file)) {
    if (!args.step_signal_dir.empty()) {
      std::string signal = args.step_signal_dir + "/step_" + std::to_string(step);
      while (!exists(args.stop_file) && !exists(signal)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
      if (exists(args.stop_file)) {
        break;
      }
      issue_burst();
      ++step;
      continue;
    }

    if (interval_ns > 0) {
      long long delay_ns = next_burst_ns - now_ns();
      if (delay_ns > 0) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(delay_ns));
      }
      next_burst_ns += interval_ns;
    }
    issue_burst();
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  stop_ns = now_ns();

  double elapsed_s = static_cast<double>(stop_ns - start_ns) / 1e9;
  double achieved_gbps =
      elapsed_s > 0 ? (static_cast<double>(bytes) / elapsed_s) / 1e9 : 0.0;
  double chunks_per_s =
      elapsed_s > 0 ? static_cast<double>(chunks) / elapsed_s : 0.0;
  std::printf("[lmcache_kv_runner] stopped bursts=%llu chunks=%llu bytes=%llu "
              "elapsed_s=%.6f achieved_gbps=%.3f chunks_per_s=%.3f\n",
              bursts, chunks, bytes, elapsed_s, achieved_gbps, chunks_per_s);
  std::fflush(stdout);

  if (!args.stats_out.empty()) {
    std::ofstream out(args.stats_out);
    out << "{\n"
        << "  \"stats\": [{\n"
        << "    \"direction\": \"lmcache_kv_cuMemcpyDtoDAsync\",\n"
        << "    \"src_device\": " << args.src_device << ",\n"
        << "    \"dst_device\": " << args.dst_device << ",\n"
        << "    \"chunk_bytes\": " << args.chunk_bytes << ",\n"
        << "    \"chunks_per_burst\": " << args.chunks_per_burst << ",\n"
        << "    \"burst_interval_us\": " << args.burst_interval_us << ",\n"
        << "    \"step_synchronized\": "
        << (args.step_signal_dir.empty() ? "false" : "true") << ",\n"
        << "    \"bursts\": " << bursts << ",\n"
        << "    \"chunks\": " << chunks << ",\n"
        << "    \"bytes\": " << bytes << ",\n"
        << "    \"elapsed_s\": " << elapsed_s << ",\n"
        << "    \"achieved_gbps\": " << achieved_gbps << ",\n"
        << "    \"chunks_per_s\": " << chunks_per_s << "\n"
        << "  }]\n"
        << "}\n";
  }

  for (auto& event : events) {
    cudaEventDestroy(event);
  }
  cudaStreamDestroy(stream);
  cudaSetDevice(args.src_device);
  cudaFree(src);
  cudaSetDevice(args.dst_device);
  cudaFree(dst);
  return 0;
}
