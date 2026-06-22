```
sudo -E env \
  PATH="/home/wxzheng/miniconda3/envs/vllm-nvlink/bin:$PATH" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  nsys profile \
    --gpu-metrics-devices=0,1 \
    --gpu-metrics-frequency=10000 \
    --cuda-graph-trace=node \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    -o ../results/nsys_kv_layout/b600_s2048_contiguous \
  env NPROC=2 bash run.sh \
    --tp 1 --dp 2 \
    --batch-size 600 \
    --seq-len 2048 \
    --max-num-seqs 600 \
    --max-decode-steps 2 \
    --gpu-memory-utilization 0.95 \
    --block-layout contiguous \
    --iters 5 \
    --warmup-iters 2 \
    --output-json ../results/nsys_kv_layout/b600_s2048_contiguous.json
```


```
NPROC=4 bash run.sh   --tp 1 --dp 2   --batch-size 600   --seq-len 2048   --max-num-seqs 600   --max-decode-steps 2   --gpu-memory-utilization 0.95   --iters 50   --warmup-iters 10   --lmcache-kv-traffic   --lmcache-prefill-ranks 2   --lmcache-direction egress   --lmcache-chunk-bytes 50331648   --lmcache-chunks-per-burst 128   --skip-final-barrier   --hard-exit-after-write   --output-json results/dp_ep2_lmcache_kv_egress/b600_s2048.json
```