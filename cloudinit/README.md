# llama.cpp server — Qwen3.6-27B Fable-Fusion (80GB GPU)

Boots an [OpenAI-compatible](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
`llama-server` in Docker, serving the
[DavidAU Qwen3.6-27B Fable-Fusion](https://huggingface.co/DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF)
GGUF (Q8_0, ~30 GB) with the vision `mmproj` so it accepts images too. Protected
by a bearer token. Tuned for an 80 GB card (A100/H100); runs the whole model on
GPU with a 64k context (~34 GB VRAM measured, native max 262144).

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
- Model ID: `fable-fusion-27b`
- Context: 65536 (image input supported — `mmproj` is loaded)

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
- **GPU-in-container check** runs before the ~30 GB download, and reconfigures
  the NVIDIA runtime + restarts Docker once if the GPU isn't visible.
- **Re-runnable.** Completed, size-verified files are skipped; interrupted
  downloads resume.
- If the MTP quant misbehaves (token acceptance <50%), swap `MODEL_FILE` to the
  commented non-MTP line (same size) and re-run.
