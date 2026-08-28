# llama.cpp server — Qwen3.8-Flash-Next Uncensored (multi-GPU)

Boots an [OpenAI-compatible](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
`llama-server` serving
[orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF)
(**Q5_K_M**, ~125 GiB, split GGUF) plus the F16 vision projector. Protected by a
bearer token.

This is a **177B-param MoE** (`qwen4exp`: 512 experts, ~6B active, plus a ~51B
PLE n-gram table). It is **not** the 27B Qwen 3.8 in [`../qwen-3.8/`](../qwen-3.8/).
A single 80 GB card cannot hold 5-bit weights.

## Spheron GPU: cheapest that can run 5-bit

**There is no 6-bit GGUF.** Q6_K / Q8 would make the PLE tensor larger than
Hugging Face’s 50 GB per-file cap, so this repo tops out at **Q5_K_M**.

Weights + projector:

| Quant | Disk | Notes |
|---|---|---|
| **Q5_K_M** (default) | **125 GiB** (134 GB decimal, 3 parts) | Highest fidelity in this repo |
| Q5_K_S | 119 GiB | Slightly smaller 5-bit |

Budget **file size + KV + CUDA overhead**. Unsloth’s 5-bit row is ~200 GB *total*
memory (RAM+VRAM) if you include runtime state.

Live Spheron on-demand (Aug 2026, [pricing](https://www.spheron.network/pricing/),
per GPU). You need **~150 GiB+ VRAM for all-GPU Q5_K_M**, or **~80 GiB VRAM +
≥64 GB host RAM** if the PLE table is pinned to CPU:

| Spheron box | VRAM | Host RAM (listed SKU) | All-GPU Q5_K_M? | ~$/hr | Verdict |
|---|---|---|---|---|---|
| **2× A100 80GB** | 160 GB | 480 GB | **Yes** | **~$2.86** | **Best cheap default** |
| 1× RTX PRO 6000 96GB | 96 GB | 360 GB | No — PLE → RAM | ~$2.31 | Cheapest *if* you accept CPU PLE + shorter context |
| 3× L40S 48GB | 144 GB | 72 GB | Tight; RAM is low for PLE | ~$2.88 | Skip unless the SKU has more RAM |
| 1× H200 141GB | 141 GB | 1800 GB | Tight; script will PLE→RAM | ~$4.79 | Simpler single GPU, not cheaper |
| 2× H100 80GB | 160 GB | 180 GB | Yes | ~$5.30 | Faster than A100, not cheaper |
| 1× B200 192GB | 192 GB | 2048 GB | Yes | ~$7.20 | Easy, expensive |
| 1× A100 80GB | 80 GB | 480 GB | No | ~$1.43 | Will OOM at Q5 even with PLE offload |
| RTX 4090 24GB / 5090 32GB | 24–32 GB | — | No | $0.53–$0.86 | Need 5–6 cards and still a bad fit |

**Rent 2× A100 80GB on Spheron.** Same family as the 27B box, ~$2.86/hr, 160 GB
VRAM. The OS disk is only **~96 GB**; the large NVMe is **`/ephemeral` (~1.5 TB)**.
The installer puts models and Docker there. Do not download GGUFs to `/`.

Do **not** use a 1× 80 GB A100 for this quant. The 27B Q8_0 setup does not apply.

The installer pins `per_layer_token_embd.weight` to CPU when total VRAM is under
~150 GiB (PRO 6000, H200, single A100). On 2×80GB it stays on GPU.

## llama.cpp version

`qwen4exp` landed in llama.cpp **2026-08-27**
([PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)). Older
`ghcr.io/ggml-org/llama.cpp:server-cuda` images fail with `unknown architecture
'qwen4_exp'`. This script **builds llama-server from llama.cpp master** in Docker
(`BUILD_LLAMA=1`). Set `BUILD_LLAMA=0` to pull GHCR instead once you know that
tag is new enough. Host CUDA toolkit is still not required.

## Hugging Face token (required)

This repo is **gated**. Anonymous `curl` gets **HTTP 401**. Before install:

1. Open [the model page](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) while logged in and **accept the access terms**.
2. Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. On the GPU box:

```bash
printf '%s\n' 'hf_YOUR_TOKEN' > /root/hf-token.txt
chmod 600 /root/hf-token.txt
```

A token without accepting the terms still 401s.

`curl: (23) Failure writing output` on this Spheron SKU means **`/` is full**.
`/dev/vda1` is ~96 GB; the 1.5 TB disk is `/ephemeral`. If that already happened:

```bash
pkill -f llama-setup.sh || true
# keep the large partials — do not rm the 40GB files
curl -fsSL https://raw.githubusercontent.com/spydrful/cloudinit-llama/cursor/qwen38-flash-next-3063/cloudinit/qwen-3.8-flash-next/bash_setup.sh | bash
tail -f /root/llama-setup.log
```

The updated script moves `/models` and Docker’s data-root to `/ephemeral` and resumes part 2.

## Deploy (Spheron: use bash)

```bash
# as root — Ubuntu 22.04/24.04 or AlmaLinux / Rocky / RHEL 8–10
# 2× A100 80GB, ~200 GB free disk, /root/hf-token.txt already written
curl -fsSL https://raw.githubusercontent.com/spydrful/cloudinit-llama/cursor/qwen38-flash-next-3063/cloudinit/qwen-3.8-flash-next/bash_setup.sh | bash
tail -f /root/llama-setup.log
```

Wait for `DONE. API key:`. First boot may install an NVIDIA driver and reboot
once; SSH back in and tail the same log. Compile + 125 GiB download takes a
while.

Q5_K_S instead of Q5_K_M: edit `QUANT=Q5_K_S` in `/usr/local/sbin/llama-setup.sh`
and re-run it.

## API

```bash
sudo cat /root/llama-api-key.txt
curl -H "Authorization: Bearer <token>" http://<instance-ip>:8080/v1/models
```

- Base URL: `http://<instance-ip>:8080/v1`
- Model ID: `qwen3.8-flash-next-uncensored`
- Context default in the script: **65536** (native 262144 — raise `CTX` if VRAM
  allows)
- Sampling (thinking): temp 1.0, top_p 0.95, top_k 20, min_p 0.0
- OpenCode `reasoning_effort`: `low` / `medium` / `xhigh` (not `high`)

OpenCode snippet for a second provider alongside the 27B A100:

```json
"llama-flash": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "Qwen 3.8 Flash-Next",
  "options": {
    "baseURL": "http://<flash-next-ip>:8080/v1",
    "apiKey": "<token>"
  },
  "models": {
    "qwen3.8-flash-next-uncensored": {
      "name": "Qwen3.8-Flash-Next Uncensored",
      "id": "qwen3.8-flash-next-uncensored",
      "limit": { "context": 65536, "output": 16384 },
      "modalities": { "input": ["text", "image"] },
      "options": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "chat_template_kwargs": {
          "enable_thinking": true,
          "reasoning_effort": "medium"
        }
      }
    }
  }
}
```

## Logs

```bash
sudo docker logs -f llama
```

## Notes

- Point llama.cpp at the **`…-00001-of-00003.gguf`** part; it loads the rest.
- No MTP head in these GGUFs.
- `cloud-config.init` is not provided — Spheron’s user-data field is often a
  bash runner, not cloud-init YAML.
