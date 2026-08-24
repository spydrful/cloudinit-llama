# llama.cpp server — Qwen3.8-27B Uncensored (80GB GPU)

Boots an [OpenAI-compatible](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
`llama-server` in Docker, serving the
[JonathanColetti Qwen3.8-27B Uncensored](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF)
GGUF (Q8_0, ~29 GB) with the vision projector so it accepts images too. Protected
by a bearer token. Tuned for an 80 GB card (A100/H100); runs the whole model on
GPU with **524288** context on one 80GB card (2× YaRN) or **1048576** on two (4× YaRN).

Targets NVIDIA + Debian-based hosts (Ubuntu 22.04/24.04). Docker image is
`ghcr.io/ggml-org/llama.cpp:server-cuda` (built on CUDA 12.8, runs on newer
drivers via forward compatibility).

## Two deploy forms — pick one

### `cloud-config.init` — real cloud-init
Paste into the provider's **cloud-init / user-data** field. It writes and runs
the setup script on first boot.

### `bash_setup.sh` — plain bash
For providers that **don't honour cloud-init** — some silently repurpose the
"cloud-init" field as a bash runner (they `bash` your payload), so YAML fails.
This form is a straight bash script: it writes `/root/setup.sh`, which you then
run. Also the right choice when you already have a root shell:

```bash
sudo bash bash_setup.sh          # writes /root/setup.sh
sudo nohup bash /root/setup.sh > ~/llama-setup.log 2>&1 &
tail -f ~/llama-setup.log
```

## API token

Set `API_KEY` at the top of the script. Left as `CHANGE-ME`, a random token is
generated to `/root/llama-api-key.txt` (mode 600):

```bash
sudo cat /root/llama-api-key.txt
```

## Reaching it

The server listens on **:8080** — open that port in the provider firewall. All
`/v1/*` routes require the bearer token, but keep the token strong since the
port faces the internet.

Test:

```bash
curl -H "Authorization: Bearer <token>" http://<instance-ip>:8080/v1/models
```

Point any OpenAI-compatible client (e.g. Cline "OpenAI Compatible") at it:

- Base URL: `http://<instance-ip>:8080/v1`
- API key: your token
- Model ID: `qwen3.8-27b-uncensored`
- Context: **524288** on 1×80GB, **1048576** on 2×80GB (`YARN=1`). Native 262144 if `YARN=0`.

## YaRN / longer context

Qwen3.8 natively trains at **262,144**. Unsloth/Qwen document YaRN up to **1,048,576** (4×). Same GGUF; only RoPE scaling and KV size change.

llama.cpp pre-allocates KV for `-c`. Weights stay ~29 GB (Q8_0) + ~0.9 GB vision. Only 16 of 64 layers are full attention (4 KV heads × dim 256), so KV is ~64 KiB/token at f16:

| Context | YaRN | KV (f16) | Q8_0 + vision + KV | 1× A100 80GB | 2× A100 80GB (160 GB) |
|---|---|---:|---:|---|---|
| 262,144 | off | ~16 GB | ~46 GB | fits | fits |
| **524,288** | **2×** | **~32 GB** | **~62 GB** | **fits, 1-GPU default** | fits |
| 786,432 | 3× | ~48 GB | ~78 GB | too tight | fits |
| **1,048,576** | **4×** | **~64 GB** | **~94 GB** | f16 **OOM** | **fits, 2-GPU default** |

The second GPU is for the cache, not the weights. Q8_0 is only ~30 GB; 1M f16 KV is ~64 GB, so one 80GB card cannot hold both. With `--tensor-split 1,1`, each GPU holds ~15 GB weights + ~32 GB KV.

`YARN=1` picks this from `nvidia-smi` at boot: **512k on one GPU, 1M on two**. Set `YARN=0` for native 262k. Going past 1M is outside Qwen’s documented YaRN range.

Already-running **1 GPU** (Cline context **524288**):

```bash
API_KEY=$(sudo cat /root/llama-api-key.txt)
docker rm -f llama
docker run -d --name llama --restart unless-stopped --gpus all \
  -p 8080:8080 -v /models:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Qwen3.8-27B-Uncensored-Q8_0.gguf \
  --mmproj /models/Qwen3.8-27B-Uncensored-vision-f16.gguf \
  --host 0.0.0.0 --port 8080 --api-key "$API_KEY" \
  -ngl 999 -c 524288 --jinja --alias qwen3.8-27b-uncensored \
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 \
  --flash-attn on \
  --override-kv qwen35.context_length=int:524288
```

Already-running **2 GPU** (Cline context **1048576**):

```bash
API_KEY=$(sudo cat /root/llama-api-key.txt)
docker rm -f llama
docker run -d --name llama --restart unless-stopped --gpus all \
  -p 8080:8080 -v /models:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Qwen3.8-27B-Uncensored-Q8_0.gguf \
  --mmproj /models/Qwen3.8-27B-Uncensored-vision-f16.gguf \
  --host 0.0.0.0 --port 8080 --api-key "$API_KEY" \
  -ngl 999 -c 1048576 --jinja --alias qwen3.8-27b-uncensored \
  --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 \
  --flash-attn on --tensor-split 1,1 \
  --override-kv qwen35.context_length=int:1048576
```

Two cards can also run **two separate 512k servers** instead of one 1M server if you care more about concurrent jobs than a million-token window.

## Logs

```bash
sudo docker logs -f llama          # per-request tokens, tok/s, slots (LM Studio-style)
sudo docker logs --tail 100 llama
```

Add `-v` to the `docker run` line for full prompt/response bodies (verbose —
client prompts can be large).

## Notes / field lessons baked in

- **Single-stream download.** HF now serves these blobs via the Xet CDN, which
  signs each redirect for one byte range — multi-connection downloaders (aria2
  `-x16`) get a 403 storm. The script uses a single resumable `curl` with a
  size check against the exact expected byte count.
- **GPU-in-container check** runs before the ~29 GB download, and reconfigures
  the NVIDIA runtime + restarts Docker once if the GPU isn't visible.
- **Re-runnable.** Completed, size-verified files are skipped; interrupted
  downloads resume.
- Default quant is the fused MTP Q8_0. If the MTP quant misbehaves (token
  acceptance <50%), swap `MODEL_FILE` / `MODEL_SIZE` to the commented noMTP
  lines and re-run.
