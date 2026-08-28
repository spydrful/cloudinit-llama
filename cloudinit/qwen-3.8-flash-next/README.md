# llama.cpp server — Qwen3.8-Flash-Next Uncensored (multi-GPU)

Boots an [OpenAI-compatible](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
`llama-server` serving
[orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF)
(**Q5_K_M**, ~125 GiB, split GGUF) plus the F16 vision projector. Protected by a
bearer token.

This is a **177B-param MoE** (`qwen4exp`: 512 experts, ~6B active, plus a ~51B
PLE n-gram table). It is **not** the 27B Qwen 3.8 in [`../qwen-3.8/`](../qwen-3.8/).
A single 80 GB card cannot hold 5-bit weights. **2× A100** holds Q5_K_M at **64k**
context. **4× A100** is what you want for native **262144**.

## Spheron GPU: 4× A100 for 262k context

**There is no 6-bit GGUF.** Q6_K / Q8 would make the PLE tensor larger than
Hugging Face’s 50 GB per-file cap, so this repo tops out at **Q5_K_M**.

Weights + projector:

| Quant | Disk | Notes |
|---|---|---|
| **Q5_K_M** (default) | **125 GiB** (134 GB decimal, 3 parts) | Highest fidelity in this repo |
| Q5_K_S | 119 GiB | Slightly smaller 5-bit |

Budget **file size + KV + CUDA overhead**. Q5_K_M weights are ~125 GiB. Native
262k KV needs the extra cards, not a bigger quant.

| Spheron box | VRAM | CTX this script uses | ~$/hr | Verdict |
|---|---|---|---|---|
| **4× A100 80GB** | 320 GB | **262144** | **~$5.72** | **Default for long OpenCode sessions** |
| 2× A100 80GB | 160 GB | auto-cap **65536** | ~$2.86 | Fits weights; 262k KV OOMs |
| 1× A100 80GB | 80 GB | — | ~$1.43 | OOM even at Q5 |
| 2× H100 80GB | 160 GB | 65536 | ~$5.30 | Same VRAM as 2× A100, not 262k |
| 4× H100 80GB | 320 GB | 262144 | ~$10.60 | Faster than 4× A100, not cheaper |

**Rent 4× A100 80GB on Spheron** (~4 × $1.43/hr). PCIe is fine; llama.cpp splits
layers across GPUs (`--gpus all`). The OS disk is still **~96 GB**; the large
NVMe is **`/ephemeral`**. The installer puts models and Docker there.

The installer pins `per_layer_token_embd.weight` to CPU when total VRAM is under
~150 GiB. On 2× or 4× 80GB it stays on GPU.

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
# 4× A100 80GB for 262k, /root/hf-token.txt already written
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
- Context default: **262144** on 4×80GB (auto-caps to **65536** on 2×80GB)
- Sampling (thinking): temp 1.0, top_p 0.95, top_k 20, min_p 0.0
- OpenCode `reasoning_effort`: `low` / `medium` / `xhigh` (not `high`)

OpenCode snippet for a second provider alongside the 27B A100:

```json
"llama-flash": {
  "npm": "@ai-sdk/openai-compatible",
      "name": "Qwen 3.8 Flash-Next (4x A100)",
  "options": {
    "baseURL": "http://<flash-next-ip>:8080/v1",
    "apiKey": "<token>"
  },
  "models": {
    "qwen3.8-flash-next-uncensored": {
      "name": "Qwen3.8-Flash-Next Uncensored",
      "id": "qwen3.8-flash-next-uncensored",
          "limit": { "context": 262144, "output": 32768 },
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
