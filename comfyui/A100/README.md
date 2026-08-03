# ComfyUI + MiniMax H3 (A100 80GB)

Native ComfyUI install serving [MiniMax H3](https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui) —
an open-weights omni-modal video model — tuned for a single **A100 80GB** on
**Ubuntu 24.04 LTS + CUDA 13.0 + Docker** host.

MiniMax H3 does text→video, image→video, first/last-frame control, and
reference→video (carry a subject/motion/voice from up to **9 reference images,
3 reference videos, 3 audio clips**). Output up to 2K and 15 seconds, with
native audio.

## Why native, not Docker

The model is day-0 (Aug 2026) and needs **ComfyUI ≥ 0.30.0** — newer than any
published Docker image. The script clones ComfyUI from git and installs
PyTorch **cu128** wheels, which run on the CUDA-13 driver via forward
compatibility (the wheels bundle their own CUDA runtime; the driver only has to
be new enough). This is the standard, most reliable ComfyUI deployment.

## A100 tuning

A100 is Ampere — **INT8 tensor cores are native**, FP8/FP4 are not. The script
pulls the `pruned_int8_convrot` diffusion models (~21 GB each), which run
natively on Ampere. Weights total ~63 GB, leaving ~17 GB+ of the 80 GB card for
video activations.

| File | Folder | Size |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `diffusion_models` | 21.0 GB |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `diffusion_models` | 21.0 GB |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `text_encoders` | 15.7 GB |
| `minimax_h3_video_vae_fp16.safetensors` | `vae` | 5.2 GB |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae` | 0.6 GB |

The text encoder defaults to `nvfp4_awq` because that is what the official
templates reference and it runs on any GPU. For maximum Ampere-native quality
you can switch to `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (27 GB) —
flip the two `TEXT_ENCODER` / `TE_SIZE` lines at the top of `setup.sh`, then
re-point the text-encoder loader dropdown in the workflow to that file.

## Requirements

- A100 80GB, Ubuntu 24.04, recent NVIDIA driver (CUDA 13 / driver 580 tested).
- **~80 GB free disk** on whatever holds `COMFY_DIR` (~63 GB models + ~8 GB torch).

## Run

```bash
sudo nohup bash setup.sh > ~/comfy-setup.log 2>&1 &
tail -f ~/comfy-setup.log
```

First run downloads ~63 GB of weights and installs PyTorch — budget 15–40 min
depending on network. Re-running is safe: completed, size-verified files are
skipped and the download resumes where it stopped.

Logs once running (systemd, journald):

```bash
journalctl -u comfyui -f
```

## Security — read before exposing

ComfyUI has **no authentication and can execute arbitrary code** (custom nodes,
model loading). **Do not** bind it to `0.0.0.0` on the public internet. The
script binds `127.0.0.1` on purpose. Reach it from your machine over an SSH
tunnel:

```bash
ssh -L 8188:localhost:8188 root@<instance-ip>
```

Then open <http://localhost:8188> — encrypted and private. For tunnel-free
browser access, put a basic-auth reverse proxy (Caddy/nginx) in front and
restrict source IPs at the provider firewall.

## Using it

In the browser (over the tunnel):

1. **Template Library → Video → MiniMax H3 T2V / I2V / R2V**. Models are already
   in place, so the templates load ready to run.
2. **T2V**: type a prompt → Run. **I2V**: drag an image (optional first/last
   frame) → Run. **R2V**: drop reference images / videos / audio → Run.
3. Output video lands in `ComfyUI/output/`, downloadable straight from the video
   node.

Programmatic ("send text/image, get video back" from code): in the GUI use
**Save (API Format)** on a template to export its workflow JSON, then
`POST /prompt` → poll `/history/{id}` → `GET /view` for the file. Same port,
through the tunnel.

First run loads ~63 GB of weights (slow) and 2K/15 s is heavy even on an A100 —
validate with a short low-res clip first, then scale up.

## Sources

- [ComfyUI blog: MiniMax H3 day-0 support](https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui)
- [Official setup docs](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [Comfy-Org/MiniMax-H3 weights](https://huggingface.co/Comfy-Org/MiniMax-H3)
