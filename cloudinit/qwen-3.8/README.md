# llama.cpp server — Qwen3.8-27B Uncensored (80GB GPU)

Boots an [OpenAI-compatible](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
`llama-server` in Docker, serving the
[JonathanColetti Qwen3.8-27B Uncensored](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF)
GGUF (Q8_0, ~29 GB) with the vision projector so it accepts images too. Protected
by a bearer token. Tuned for an 80 GB card (A100/H100); runs the whole model on
GPU with **524288** context (2× YaRN). Tuned for a typical rental: **1× A100 80GB, 28 vCPU, 120 GB RAM**.

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
This form is a straight bash script: it writes `/root/setup.sh` and starts it
in the background. Also the right choice when you already have a root shell
(including `curl | bash`):

```bash
curl -fsSL https://raw.githubusercontent.com/spydrful/cloudinit-llama/main/cloudinit/qwen-3.8/bash_setup.sh | bash
tail -f /root/llama-setup.log
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
- Context: **524288** with 2× YaRN (`YARN=1`). Native 262144 if `YARN=0`.

## YaRN / longer context

Default is **`YARN=1` → 524288** (2× YaRN, f16 KV ~32 GB, ~62 GB VRAM with Q8_0 + vision). That is the largest comfortable window on one A100 80GB. Native 262144 if `YARN=0`. 1M needs a second 80GB card.

## Performance flags (28 vCPU / 120 GB RAM / 1× A100 80GB)

Long OpenCode sessions were re-prefilling from scratch because llama.cpp opened several slots (LRU) and skipped prompt-cache saves over the default **8 GB** `--cache-ram`. This box has 120 GB host RAM, so the startup now:

- **`-np 1`** — one slot, so the next turn reuses this chat’s KV
- **`--cache-ram -1`** — no 8 GB cap on serialized prompt state (~10 GB at 150k tokens)
- **`--cache-reuse 256`** — prefix match / KV shift when the prompt is not identical
- **`--flash-attn on -b 2048 -ub 2048`** — faster prefill on 80GB
- **`--threads 8 --threads-http 4`** — GPU does the model; leave the rest of the 28 vCPUs to the OS
- **`--shm-size 16g`** on the container

Already-running box (no re-download; Cline/OpenCode context **524288**):

```bash
API_KEY=$(sudo cat /root/llama-api-key.txt)
docker rm -f llama
docker run -d --name llama --restart unless-stopped --gpus all \
  --shm-size 16g \
  -p 8080:8080 -v /models:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Qwen3.8-27B-Uncensored-Q8_0.gguf \
  --mmproj /models/Qwen3.8-27B-Uncensored-vision-f16.gguf \
  --host 0.0.0.0 --port 8080 --api-key "$API_KEY" \
  -ngl 999 -c 524288 --jinja --alias qwen3.8-27b-uncensored \
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 \
  --override-kv qwen35.context_length=int:524288 \
  --flash-attn on -np 1 \
  --cache-ram -1 --cache-reuse 256 \
  -b 2048 -ub 2048 \
  --threads 8 --threads-http 4 \
  --defrag-thold 0.1
```

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
