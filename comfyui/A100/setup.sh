#!/usr/bin/env bash
# ComfyUI + MiniMax H3 — Ubuntu 24.04, A100 80GB, CUDA 13 + Docker host (native install)
set -euxo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run as root: sudo bash $0"; exit 1; }

##### EDIT ME ##################################################
COMFY_DIR=/opt/ComfyUI       # put on your big disk; needs ~80GB free
PORT=8188
# Text encoder. Default nvfp4_awq matches the official templates + runs on any GPU.
# A100-native alt (Ampere INT8, higher quality, bigger): flip both lines below to
#   qwen3vl_32b_minimax_h3_int8_convrot.safetensors / 27141342152
# ...then re-point the loader dropdown in the workflow to that file.
TEXT_ENCODER="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
TE_SIZE=15687142551
################################################################

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git curl ca-certificates python3-venv python3-pip ffmpeg

nvidia-smi

# ComfyUI from git (latest; MiniMax H3 needs >=0.30.0)
if [ ! -d "$COMFY_DIR/.git" ]; then
  git clone https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
else
  git -C "$COMFY_DIR" pull --ff-only || true
fi

# venv + deps. torch cu128 wheels run on the CUDA-13 driver (forward-compat).
python3 -m venv "$COMFY_DIR/venv"
"$COMFY_DIR/venv/bin/pip" install --upgrade pip
"$COMFY_DIR/venv/bin/pip" install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
"$COMFY_DIR/venv/bin/pip" install -r "$COMFY_DIR/requirements.txt"

# Models — single-stream curl (HF Xet CDN 403s multi-connection tools),
# resumable, size-verified, skipped when already complete.
mkdir -p "$COMFY_DIR/models/diffusion_models" \
         "$COMFY_DIR/models/text_encoders" \
         "$COMFY_DIR/models/vae"
REPO=Comfy-Org/MiniMax-H3
dl() { # subpath expected_bytes destdir
  local f="$3/$(basename "$1")"
  if [ -f "$f" ] && [ "$(stat -c%s "$f")" -eq "$2" ]; then return 0; fi
  curl -fL --retry 10 --retry-delay 5 -C - -o "$f" \
    "https://huggingface.co/${REPO}/resolve/main/$1"
  [ "$(stat -c%s "$f")" -eq "$2" ] || { echo "SIZE MISMATCH: $f"; exit 1; }
}
dl "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" 20970379616 "$COMFY_DIR/models/diffusion_models"
dl "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" 20970379616 "$COMFY_DIR/models/diffusion_models"
dl "text_encoders/${TEXT_ENCODER}" "$TE_SIZE" "$COMFY_DIR/models/text_encoders"
dl "vae/minimax_h3_video_vae_fp16.safetensors" 5207808496 "$COMFY_DIR/models/vae"
dl "vae/minimax_h3_audio_vae_fp32.safetensors" 605254808 "$COMFY_DIR/models/vae"

# Service — binds LOCALHOST only (ComfyUI has no auth; see note). Auto-restarts.
cat > /etc/systemd/system/comfyui.service <<UNIT
[Unit]
Description=ComfyUI
After=network.target

[Service]
WorkingDirectory=$COMFY_DIR
ExecStart=$COMFY_DIR/venv/bin/python main.py --listen 127.0.0.1 --port $PORT
Restart=always
User=root

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now comfyui
echo "DONE. ComfyUI on 127.0.0.1:$PORT"
