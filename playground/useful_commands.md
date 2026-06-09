```
NIXL_FAKE_READ=1 \
bash playground/moe_pd/start_server.sh
```

Use nsys instead of torch profiler
```
VLLM_PROFILE_KIND=cuda \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --trace=cuda,nvtx,ucx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end repeat \
  --force-overwrite=true \
  -o playground/log/nsys_pd/pd_kv \
  bash playground/moe_pd/start_server.sh
```

Nsys with GPU counters
```
sudo -E env "PATH=$PATH" \
  VLLM_PROFILE_KIND=cuda \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" \
  nsys profile \
    --gpu-metrics-device=all \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end repeat \
    --force-overwrite=true \
    -o playground/log/nsys_pd/pd_kv \
    bash playground/moe_pd/start_server.sh
```

Minimal nsys for CUDA graph and MemCpy correlation.
```
sudo -E env "PATH=$PATH"   VLLM_PROFILE_KIND=cuda   VLLM_WORKER_MULTIPROC_METHOD=spawn   "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"   nsys profile     --gpu-metrics-device=all     --trace-fork-before-exec=true     --cuda-graph-trace=graph --cuda-memory-usage=false --cudabacktrace=none     --trace=cuda,nvtx     --sample=none     --cpuctxsw=none     --capture-range=cudaProfilerApi     --capture-range-end repeat     --force-overwrite=true     -o playground/log/nsys_pd/pd_kv     bash playground/moe_pd/start_server.sh
```


Optional logging for UCX and NIXL
```
UCX_LOG_LEVEL=debug
NIXL_LOG_LEVEL=DEBUG
```

Random dataset
```
vllm bench serve   --backend openai-chat   --base-url http://127.0.0.1:8000   --endpoint /v1/chat/completions   --model Qwen3-30B-A3B-Instruct-2507   --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507   --dataset-name random   --random-input-len 1024   --random-output-len 64   --num-prompts 50   --request-rate 5 --burstiness inf --seed 0 --save-detailed --save-result --plot-timeline --result-dir playground/out/vllm_bench_pd --result-filename pd_bench.json
```

ShareGPT 
```
vllm bench serve   --backend openai-chat   --base-url http://127.0.0.1:8000   --endpoint /v1/chat/completions   --model Qwen3-30B-A3B-Instruct-2507   --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507   --dataset-name sharegpt --dataset-path datasets/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 500 --request-rate 20 --save-detailed --save-result --plot-timeline --result-dir playground/out/vllm_bench_pd --result-filename pd_bench.json
```

Supported models
```
Qwen/Qwen3-30B-A3B-Instruct-2507
Qwen/Qwen3-235B-A22B-Instruct-2507
```

Track hanging vllm Engines's parent
`pstree -spT PID`