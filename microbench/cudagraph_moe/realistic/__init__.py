"""Realistic decode microbenchmark for vLLM's CUDA-graph path.

Sibling to ``base/`` — same model + same captured graphs, but the
benchmark loop programs per-request seq_lens, block tables, and slot
mappings directly so that:

  - per-request prefill length varies (heterogeneous KV depths),
  - per-request "age" (decode step count) varies,
  - the captured graph's ``reshape_and_cache`` actually writes KV.

See ``README.md`` for the full code-path walkthrough.
"""
