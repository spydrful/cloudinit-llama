# llama.cpp cloud-init setups

Each subfolder is a self-contained `llama.cpp` OpenAI-compatible server for one
model (cloud-init **and** plain-bash forms). Pick one:

| Folder | Model |
|---|---|
| [`qwen-3.6/`](qwen-3.6/) | Qwen3.6-27B Fable-Fusion GGUF (Q8_0) with vision |
| [`qwen-3.8/`](qwen-3.8/) | Qwen3.8-27B Uncensored GGUF (Q8_0) with vision |
| [`qwen-3.8-flash-next/`](qwen-3.8-flash-next/) | Qwen3.8-Flash-Next Uncensored GGUF (Q5_K_M, ~125 GiB MoE; **4× 80 GB** for 1M YaRN) |
