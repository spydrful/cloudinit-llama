# llama.cpp server — GLM-5.3-Flash IQ3 (2× A100)

Boots an [OpenAI-compatible](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
`llama-server` serving
[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)
**UD-IQ3_XXS** (~120 GB, 4 split parts) plus `mmproj-F16.gguf` (~1.1 GB). Bearer
token on `:8080`.

This is Z.ai’s **320B MoE / 18B active** multimodal GLM-5.3-Flash (`glm5next`),
not the 27B Qwen boxes and not Flash-Next Q5. Unsloth’s 3-bit is the quant that
fits **2× A100 80GB** (160 GB VRAM). Q4_K_XL (~200 GB) needs 4×80GB.

## Hardware

| Quant | Disk | Unsloth RAM+VRAM | This installer |
|---|---|---|---|
| UD-IQ3_XXS | **~120 GB** (4 parts) | 128–150 GB | **default — 2× A100** |
| UD-Q4_K_XL | ~200 GB | 162–210 GB | not this folder |

Spheron: rent **2× A100 80GB** (~$2.86/hr). OS disk is **~96 GB**; the 1.5 TB
volume is **`/ephemeral`**. The script puts models and Docker there.

Stock `ghcr.io/ggml-org/llama.cpp:server-cuda` **cannot** load `glm5next`. The
script builds [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp)
branch **`glm5next/upstream`**.

## Deploy (Spheron: bash)

```bash
# as root — Ubuntu 22.04/24.04 or AlmaLinux / Rocky / RHEL 8–10
# 2× A100 80GB, /ephemeral mounted
curl -fsSL https://raw.githubusercontent.com/spydrful/cloudinit-llama/cursor/glm53-flash-a100-3063/cloudinit/glm-5.3-flash/bash_setup.sh | bash
tail -f /root/llama-setup.log
```

Wait for `DONE. API key:`. First boot may install an NVIDIA driver and reboot
once. Compile + ~120 GB download takes a while. Part `00001-of-00004` is only
~9 MB (header shard); llama.cpp still wants all four files next to it.

If HF rate-limits you:

```bash
printf '%s\n' 'hf_YOUR_TOKEN' > /root/hf-token.txt
chmod 600 /root/hf-token.txt
bash /usr/local/sbin/llama-setup.sh >> /root/llama-setup.log 2>&1 &
```

`curl: (23)` means `/` is full — re-run this installer (it relocates to
`/ephemeral`). Do not delete large partials.

## API

```bash
sudo cat /root/llama-api-key.txt
curl -H "Authorization: Bearer <token>" http://<instance-ip>:8080/v1/models
```

- Base URL: `http://<instance-ip>:8080/v1`
- Model ID: `glm-5.3-flash`
- Context default: **65536** (native 1M — raise `CTX` if VRAM allows)
- Sampling: temp **1.0**, top_p **0.95**
- Reasoning: `low` / `high` / `max` via `chat_template_kwargs.reasoning_effort`
  (script default **high**; Unsloth’s own default is `max`)

OpenCode provider:

```json
"llama-glm": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "GLM-5.3-Flash (2x A100)",
  "options": {
    "baseURL": "http://<ip>:8080/v1",
    "apiKey": "<token>"
  },
  "models": {
    "glm-5.3-flash": {
      "name": "GLM-5.3-Flash IQ3",
      "id": "glm-5.3-flash",
      "limit": { "context": 65536, "output": 16384 },
      "modalities": { "input": ["text", "image"] },
      "options": {
        "temperature": 1.0,
        "top_p": 0.95,
        "chat_template_kwargs": { "reasoning_effort": "high" }
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

- Point llama.cpp at **`…-00001-of-00004.gguf`**; it loads 00002–00004 from the
  same directory.
- Do not run this on the same box as Flash-Next Q5 at the same time (both want
  the full 160 GB).
- `cloud-config.init` is not provided — Spheron user-data is often a bash
  runner, not cloud-init YAML.
