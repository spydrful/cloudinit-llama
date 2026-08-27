#!/usr/bin/env bash
# Writes /root/setup.sh and starts it in the background.
# Safe for:  curl -fsSL …/bash_setup.sh | bash
set -euo pipefail

cat > /root/setup.sh <<'SETUP_EOF'
#!/usr/bin/env bash
set -euxo pipefail

##### EDIT ME ##################################################
API_KEY="CHANGE-ME"          # if left as-is, random one -> /root/llama-api-key.txt
PORT=8080
CTX=262144                   # used when YARN=0 (native, no RoPE scaling)
YARN=1                       # 1 = 2x YaRN to 524288 (fits 1x A100 80GB + 120GB host RAM)
################################################################

REPO="JonathanColetti/Qwen3.8-27B-Uncensored-GGUF"
MODEL_FILE="Qwen3.8-27B-Uncensored-Q8_0.gguf"
MODEL_SIZE=29047084448
# fallback if MTP quant misbehaves (no MTP tensors; slightly smaller):
# MODEL_FILE="Qwen3.8-27B-Uncensored-noMTP-Q8_0.gguf"
# MODEL_SIZE=28595763680
MMPROJ_FILE="Qwen3.8-27B-Uncensored-vision-f16.gguf"
MMPROJ_SIZE=927606912

if [ -f /root/llama-api-key.txt ]; then
  API_KEY=$(cat /root/llama-api-key.txt)
elif [ "$API_KEY" = "CHANGE-ME" ]; then
  API_KEY=$(openssl rand -hex 24)
fi
echo "$API_KEY" > /root/llama-api-key.txt
chmod 600 /root/llama-api-key.txt

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y curl ca-certificates gnupg openssl

nvidia-smi >/dev/null 2>&1 || {
  if [ -f /root/llama-nvidia-install.attempted ]; then
    echo "nvidia-smi still failing after driver install + reboot" >&2
    lspci -nn | grep -i nvidia || true
    exit 1
  fi
  touch /root/llama-nvidia-install.attempted
  apt-get install -y build-essential dkms "linux-headers-$(uname -r)" ubuntu-drivers-common
  cat >/etc/modprobe.d/blacklist-nouveau.conf <<'NOUVEAU'
blacklist nouveau
options nouveau modeset=0
NOUVEAU
  update-initramfs -u || true
  ubuntu-drivers autoinstall || {
    . /etc/os-release
    case "${VERSION_ID:-22.04}" in
      24.04) CUDA_DISTRO=ubuntu2404 ;;
      *)     CUDA_DISTRO=ubuntu2204 ;;
    esac
    curl -fsSL -o /tmp/cuda-keyring.deb \
      "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update
    apt-get install -y cuda-drivers
  }
  cat >/etc/systemd/system/llama-setup-continue.service <<UNIT
[Unit]
Description=Continue llama.cpp setup after NVIDIA driver install
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=$(readlink -f "$0")
TimeoutStartSec=infinity
StandardOutput=append:/root/llama-setup.log
StandardError=append:/root/llama-setup.log

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable llama-setup-continue.service
  echo "NVIDIA driver installed; rebooting. Setup continues from /root/llama-setup.log"
  reboot
}
nvidia-smi
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh
systemctl enable --now docker

if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

docker run --rm --gpus all ghcr.io/ggml-org/llama.cpp:server-cuda --version || {
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
  docker run --rm --gpus all ghcr.io/ggml-org/llama.cpp:server-cuda --version
}

mkdir -p /models
dl() { # filename expected_bytes
  local f="/models/$1"
  if [ -f "$f" ] && [ "$(stat -c%s "$f")" -eq "$2" ]; then return 0; fi
  curl -fL --retry 10 --retry-delay 5 -C - -o "$f" \
    "https://huggingface.co/${REPO}/resolve/main/$1"
  [ "$(stat -c%s "$f")" -eq "$2" ] || { echo "SIZE MISMATCH: $f"; exit 1; }
}
dl "$MODEL_FILE" "$MODEL_SIZE"
dl "$MMPROJ_FILE" "$MMPROJ_SIZE"

# Typical box: 1x A100 80GB, 28 vCPU, 120GB RAM.
# -np 1 keeps OpenCode on one slot so KV prefix is reused.
# --cache-ram -1: default 8GB skip-saves ~10GB long-session states (host RAM, not VRAM).
# -ub 2048: faster prefill on 80GB; flash-attn keeps the compute buffer in check.
NPROC=$(nproc)
THREADS=$(( NPROC > 8 ? 8 : NPROC ))
HTTP_THREADS=$(( NPROC > 4 ? 4 : NPROC ))

YARN_ARGS=()
if [ "$YARN" = "1" ]; then
  CTX=524288
  # override-kv: llama-server otherwise caps slots at n_ctx_train (262144).
  YARN_ARGS=(
    --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144
    --override-kv qwen35.context_length=int:524288
  )
fi

docker rm -f llama 2>/dev/null || true
docker run -d --name llama --restart unless-stopped --gpus all \
  --shm-size 16g \
  -p "${PORT}:8080" -v /models:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m "/models/${MODEL_FILE}" \
  --mmproj "/models/${MMPROJ_FILE}" \
  --host 0.0.0.0 --port 8080 \
  --api-key "$API_KEY" \
  -ngl 999 -c "$CTX" --jinja \
  --alias qwen3.8-27b-uncensored \
  --flash-attn on -np 1 \
  --cache-ram -1 \
  -b 2048 -ub 2048 \
  --threads "$THREADS" --threads-http "$HTTP_THREADS" \
  --image-min-tokens 1024 --reasoning-preserve \
  "${YARN_ARGS[@]}"

systemctl disable llama-setup-continue.service 2>/dev/null || true
rm -f /etc/systemd/system/llama-setup-continue.service
systemctl daemon-reload 2>/dev/null || true

echo "DONE. API key: $API_KEY"
SETUP_EOF

chmod 755 /root/setup.sh
echo "Wrote /root/setup.sh — starting in background (log: /root/llama-setup.log)"
nohup bash /root/setup.sh >> /root/llama-setup.log 2>&1 &
echo "pid $!"
echo "Follow with:  tail -f /root/llama-setup.log"
