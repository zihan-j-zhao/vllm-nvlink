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

struct Args {
  int src_device = -1;
  int dst_device = -1;
  double rate_gbps = 0.0;
  size_t chunk_bytes = 0;
  size_t buffer_bytes = 0;
  int max_inflight = 64;
  std::string ready_file;
  std::string go_file;
  std::string stop_file;
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

size_t parse_mb(const char* value) {
  return static_cast<size_t>(std::strtoull(value, nullptr, 10)) * 1024ull *
         1024ull;
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
    } else if (std::strcmp(argv[i], "--rate-gbps") == 0) {
      args.rate_gbps = std::atof(next());
    } else if (std::strcmp(argv[i], "--chunk-mb") == 0) {
      args.chunk_bytes = parse_mb(next());
    } else if (std::strcmp(argv[i], "--buffer-mb") == 0) {
      args.buffer_bytes = parse_mb(next());
    } else if (std::strcmp(argv[i], "--max-inflight-copies") == 0) {
      args.max_inflight = std::atoi(next());
    } else if (std::strcmp(argv[i], "--ready-file") == 0) {
      args.ready_file = next();
    } else if (std::strcmp(argv[i], "--go-file") == 0) {
      args.go_file = next();
    } else if (std::strcmp(argv[i], "--stop-file") == 0) {
      args.stop_file = next();
    } else if (std::strcmp(argv[i], "--stats-out") == 0) {
      args.stats_out = next();
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      std::exit(1);
    }
  }
  if (args.src_device < 0 || args.dst_device < 0 || args.rate_gbps <= 0 ||
      args.chunk_bytes == 0 || args.buffer_bytes == 0 ||
      args.max_inflight < 1) {
    std::fprintf(stderr, "invalid arguments\n");
    std::exit(1);
  }
  if (args.chunk_bytes > args.buffer_bytes) {
    args.chunk_bytes = args.buffer_bytes;
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

  std::printf("[bg_p2p_runner] src=cuda:%d dst=cuda:%d chunk=%zu buffer=%zu "
              "rate=%.1fGB/s max_inflight=%d\n",
              args.src_device, args.dst_device, args.chunk_bytes,
              args.buffer_bytes, args.rate_gbps, args.max_inflight);
  std::fflush(stdout);

  touch(args.ready_file);
  while (!exists(args.go_file)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  const long long period_ns = static_cast<long long>(
      (static_cast<double>(args.chunk_bytes) / (args.rate_gbps * 1e9)) * 1e9);
  long long next_issue_ns = now_ns();
  long long start_ns = now_ns();
  long long stop_ns = start_ns;
  unsigned long long ops = 0;
  unsigned long long bytes = 0;

  while (!exists(args.stop_file)) {
    if (period_ns > 0) {
      long long delay_ns = next_issue_ns - now_ns();
      if (delay_ns > 0) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(delay_ns));
      }
      next_issue_ns += period_ns;
    }

    cudaEvent_t event = events[ops % events.size()];
    if (ops >= events.size()) {
      CUDA_CHECK(cudaEventSynchronize(event));
    }
    CUDA_CHECK(cudaMemcpyPeerAsync(dst, args.dst_device, src, args.src_device,
                                   args.chunk_bytes, stream));
    CUDA_CHECK(cudaEventRecord(event, stream));
    ++ops;
    bytes += args.chunk_bytes;
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  stop_ns = now_ns();

  double elapsed_s = static_cast<double>(stop_ns - start_ns) / 1e9;
  double achieved_gbps = elapsed_s > 0 ? (static_cast<double>(bytes) / elapsed_s) / 1e9 : 0.0;
  std::printf("[bg_p2p_runner] stopped ops=%llu bytes=%llu elapsed_s=%.6f "
              "achieved_gbps=%.3f\n",
              ops, bytes, elapsed_s, achieved_gbps);
  std::fflush(stdout);

  if (!args.stats_out.empty()) {
    std::ofstream out(args.stats_out);
    out << "{\n"
        << "  \"stats\": [{\n"
        << "    \"direction\": \"p2p\",\n"
        << "    \"src_device\": " << args.src_device << ",\n"
        << "    \"dst_device\": " << args.dst_device << ",\n"
        << "    \"ops\": " << ops << ",\n"
        << "    \"bytes\": " << bytes << ",\n"
        << "    \"elapsed_s\": " << elapsed_s << ",\n"
        << "    \"achieved_gbps\": " << achieved_gbps << "\n"
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
