#!/usr/bin/env bash
# Writes the setup script and starts it in the background.
# Safe for:  curl -fsSL …/bash_setup.sh | bash
# Ubuntu 22.04/24.04 and AlmaLinux/Rocky/RHEL 8–10.
#
# Qwen3.8-Flash-Next Uncensored GGUF (qwen4exp). Needs llama.cpp from 2026-08-27+
# (PR #27742). Default quant is Q5_K_M (~125 GiB). There is no Q6 — the PLE
# tensor would exceed HF's 50 GB file cap. Target box: 2× A100 80GB.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root." >&2
  exit 1
fi

# Gated HF repo: persist a token from the environment so curl | bash works.
if [ -n "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]; then
  umask 077
  printf '%s\n' "${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}" > /root/hf-token.txt
  chmod 600 /root/hf-token.txt
fi

mkdir -p /usr/local/sbin
cat > /usr/local/sbin/llama-setup.sh <<'SETUP_EOF'
#!/usr/bin/env bash
set -euxo pipefail

##### EDIT ME ##################################################
API_KEY="CHANGE-ME"          # if left as-is, random one -> /root/llama-api-key.txt
PORT=8080
CTX=65536                    # native max is 262144; raise if VRAM allows
QUANT=Q5_K_M                 # Q5_K_M (highest) or Q5_K_S (a bit smaller)
PLE_CPU=""                   # 1 = pin n-gram table to host RAM; auto if VRAM < ~150 GiB
BUILD_LLAMA=1                # 1 = build llama-server from llama.cpp master (qwen4exp)
HF_TOKEN=""                  # required (gated repo). Or /root/hf-token.txt
################################################################

REPO="orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF"
MMPROJ_FILE="mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf"
MMPROJ_SIZE=907543296
IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"

. /etc/os-release
case "${ID:-}" in
  ubuntu|debian) PKG=apt ;;
  almalinux|rocky|rhel|centos) PKG=dnf ;;
  *) echo "Unsupported OS: ${ID:-unknown} (need Ubuntu/Debian or Alma/Rocky/RHEL)" >&2; exit 1 ;;
esac

if [ "$PKG" = "apt" ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y curl ca-certificates gnupg openssl
else
  dnf -y install dnf-plugins-core curl ca-certificates gnupg2 openssl
fi

if [ -f /root/llama-api-key.txt ]; then
  API_KEY=$(cat /root/llama-api-key.txt)
elif [ "$API_KEY" = "CHANGE-ME" ]; then
  API_KEY=$(openssl rand -hex 24)
fi
echo "$API_KEY" > /root/llama-api-key.txt
chmod 600 /root/llama-api-key.txt

if [ -z "$HF_TOKEN" ]; then
  for tokfile in /root/hf-token.txt /root/.cache/huggingface/token /root/.huggingface/token; do
    if [ -f "$tokfile" ]; then
      HF_TOKEN=$(tr -d '[:space:]' < "$tokfile")
      [ -n "$HF_TOKEN" ] && break
    fi
  done
fi
if [ -z "$HF_TOKEN" ]; then
  cat >&2 <<'ERR'
This GGUF repo is gated (HTTP 401 without a token).

1. Log in at https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF
   and accept the access terms.
2. Create a READ token: https://huggingface.co/settings/tokens
3. On this box:
     printf '%s\n' 'hf_YOUR_TOKEN' > /root/hf-token.txt
     chmod 600 /root/hf-token.txt
4. Re-run:  bash /usr/local/sbin/llama-setup.sh
ERR
  exit 1
fi

wait_for_nvidia() {
  modprobe nvidia 2>/dev/null || true
  modprobe nvidia_uvm 2>/dev/null || true
  local i
  for i in $(seq 1 90); do
    nvidia-smi >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

nvidia_ok=0
nvidia-smi >/dev/null 2>&1 && nvidia_ok=1
if [ "$nvidia_ok" -eq 0 ] && [ -f /root/llama-nvidia-install.attempted ]; then
  wait_for_nvidia && nvidia_ok=1
fi

if [ "$nvidia_ok" -eq 0 ]; then
  if [ -f /root/llama-nvidia-install.attempted ]; then
    echo "nvidia-smi still failing after driver install + reboot" >&2
    lspci -nn | grep -i nvidia || true
    journalctl -u llama-setup-continue --no-pager -n 50 || true
    exit 1
  fi
  touch /root/llama-nvidia-install.attempted
  printf '%s\n' 'blacklist nouveau' 'options nouveau modeset=0' \
    > /etc/modprobe.d/blacklist-nouveau.conf
  if [ "$PKG" = "apt" ]; then
    apt-get install -y build-essential dkms "linux-headers-$(uname -r)" ubuntu-drivers-common
    update-initramfs -u || true
    ubuntu-drivers autoinstall || {
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
  else
    dnf -y install epel-release || true
    dnf -y install gcc make tar pciutils
    dnf -y install "kernel-devel-$(uname -r)" "kernel-headers-$(uname -r)" \
      || dnf -y install kernel-devel kernel-headers
    dnf -y install dkms || true
    MAJOR="${VERSION_ID%%.*}"
    CUDA_DISTRO="rhel${MAJOR}"
    curl -fsSL -o "/etc/yum.repos.d/cuda-${CUDA_DISTRO}.repo" \
      "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/x86_64/cuda-${CUDA_DISTRO}.repo"
    dnf clean expire-cache
    dnf -y install cuda-drivers \
      || dnf -y module install nvidia-driver:latest-dkms \
      || dnf -y module install nvidia-driver:open-dkms
    dracut -f || true
  fi
  install -m 755 "$0" /usr/local/sbin/llama-setup.sh
  restorecon -v /usr/local/sbin/llama-setup.sh 2>/dev/null \
    || chcon -t bin_t /usr/local/sbin/llama-setup.sh 2>/dev/null || true
  ln -sfn /usr/local/sbin/llama-setup.sh /root/setup.sh
  cat >/etc/systemd/system/llama-setup-continue.service <<'UNIT'
[Unit]
Description=Continue llama.cpp setup after NVIDIA driver install
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'exec >>/root/llama-setup.log 2>&1; exec /bin/bash /usr/local/sbin/llama-setup.sh'
TimeoutStartSec=infinity
SyslogIdentifier=llama-setup

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable llama-setup-continue.service
  echo "NVIDIA driver installed; rebooting. Setup continues from /root/llama-setup.log"
  sync
  systemctl reboot || reboot
  exit 0
fi
nvidia-smi
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh
systemctl enable --now docker
if command -v getenforce >/dev/null && [ "$(getenforce)" != "Disabled" ]; then
  setsebool -P container_use_devices 1 || true
fi

if ! command -v nvidia-ctk >/dev/null; then
  if [ "$PKG" = "apt" ]; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
  else
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
      > /etc/yum.repos.d/nvidia-container-toolkit.repo
    dnf -y install nvidia-container-toolkit
  fi
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

if [ "$BUILD_LLAMA" = "1" ]; then
  IMAGE=llama-qwen4exp:local
  mkdir -p /tmp/llama-docker
  cat > /tmp/llama-docker/Dockerfile <<'DF'
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git cmake build-essential curl ca-certificates libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git .
RUN cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build -j"$(nproc)" --config Release --target llama-server
ENTRYPOINT ["/src/build/bin/llama-server"]
DF
  docker build -t "$IMAGE" /tmp/llama-docker
else
  docker pull "$IMAGE"
fi

docker run --rm --gpus all "$IMAGE" --version || {
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
  docker run --rm --gpus all "$IMAGE" --version
}

case "$QUANT" in
  Q5_K_M)
    MODEL_FILE="Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00001-of-00003.gguf"
    PARTS=(
      "Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00001-of-00003.gguf:44537339136"
      "Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00002-of-00003.gguf:44714628064"
      "Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00003-of-00003.gguf:44854700960"
    )
    ;;
  Q5_K_S)
    MODEL_FILE="Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf"
    PARTS=(
      "Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00001-of-00003.gguf:44953947904"
      "Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00002-of-00003.gguf:44796192320"
      "Qwen3.8-Flash-Next-Uncensored-Q5_K_S-00003-of-00003.gguf:37978625856"
    )
    ;;
  *) echo "QUANT must be Q5_K_M or Q5_K_S, got: $QUANT" >&2; exit 1 ;;
esac

mkdir -p /models
dl() { # filename expected_bytes
  local f="/models/$1"
  if [ -f "$f" ] && [ "$(stat -c%s "$f")" -eq "$2" ]; then return 0; fi
  if [ -f "$f" ]; then
    echo "incomplete $f ($(stat -c%s "$f") bytes) — deleting and restarting"
    rm -f "$f"
  fi
  curl -fL --retry 10 --retry-delay 5 --retry-connrefused -C - \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    -o "$f" \
    "https://huggingface.co/${REPO}/resolve/main/$1?download=true"
  [ "$(stat -c%s "$f")" -eq "$2" ] || { echo "SIZE MISMATCH: $f"; exit 1; }
}

hf_probe="${PARTS[0]%%:*}"
echo "Probing Hugging Face auth for ${hf_probe}"
hf_code=$(curl -sS -o /tmp/hf-probe.body -w "%{http_code}" -L --max-time 30 \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  "https://huggingface.co/${REPO}/resolve/main/${hf_probe}?download=true" \
  -r 0-0 || true)
if [ "$hf_code" != "200" ] && [ "$hf_code" != "206" ]; then
  echo "Hugging Face refused the download (HTTP ${hf_code})." >&2
  echo "Accept the model terms while logged in, then use a READ token." >&2
  echo "  https://huggingface.co/${REPO}" >&2
  echo "  printf '%s\\n' 'hf_YOUR_TOKEN' > /root/hf-token.txt" >&2
  sed -n '1,20p' /tmp/hf-probe.body >&2 || true
  exit 1
fi
rm -f /tmp/hf-probe.body
for spec in "${PARTS[@]}"; do
  dl "${spec%%:*}" "${spec##*:}"
done
dl "$MMPROJ_FILE" "$MMPROJ_SIZE"

NPROC=$(nproc)
THREADS=$(( NPROC > 8 ? 8 : NPROC ))
HTTP_THREADS=$(( NPROC > 4 ? 4 : NPROC ))

TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
NGPU=$(nvidia-smi -L | wc -l)
echo "GPUs=${NGPU} total_vram_mib=${TOTAL_VRAM_MB}"
if [ "$NGPU" -lt 1 ]; then
  echo "No NVIDIA GPU visible" >&2
  exit 1
fi

PLE_ARGS=()
# Q5_K_M weights ~128000 MiB. Keep them on GPU only when there is ~150 GiB+ VRAM
# (2×80GB). Smaller cards pin the ~45 GiB PLE n-gram table to host RAM.
if [ "${PLE_CPU}" = "1" ] || [ "$TOTAL_VRAM_MB" -lt 150000 ]; then
  PLE_ARGS=( --override-tensor 'per_layer_token_embd.weight=CPU' )
  echo "PLE n-gram table -> CPU RAM"
fi

docker rm -f llama 2>/dev/null || true
docker run -d --name llama --restart unless-stopped --gpus all \
  --shm-size 16g \
  -p "${PORT}:8080" -v /models:/models \
  "$IMAGE" \
  -m "/models/${MODEL_FILE}" \
  --mmproj "/models/${MMPROJ_FILE}" \
  --host 0.0.0.0 --port 8080 \
  --api-key "$API_KEY" \
  -ngl 999 -c "$CTX" --jinja \
  --alias qwen3.8-flash-next-uncensored \
  --flash-attn on -np 1 \
  --cache-ram -1 \
  -b 2048 -ub 2048 \
  --threads "$THREADS" --threads-http "$HTTP_THREADS" \
  --image-min-tokens 1024 --reasoning-preserve \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  "${PLE_ARGS[@]}"

if command -v firewall-cmd >/dev/null && systemctl is-active --quiet firewalld; then
  firewall-cmd --permanent --add-port="${PORT}/tcp" || true
  firewall-cmd --reload || true
fi

systemctl disable llama-setup-continue.service 2>/dev/null || true
rm -f /etc/systemd/system/llama-setup-continue.service
systemctl daemon-reload 2>/dev/null || true

echo "DONE. API key: $API_KEY"
echo "Model: $MODEL_FILE  ctx=$CTX  gpus=$NGPU"
SETUP_EOF

chmod 755 /usr/local/sbin/llama-setup.sh
ln -sfn /usr/local/sbin/llama-setup.sh /root/setup.sh
restorecon -v /usr/local/sbin/llama-setup.sh 2>/dev/null \
  || chcon -t bin_t /usr/local/sbin/llama-setup.sh 2>/dev/null || true
echo "Wrote /usr/local/sbin/llama-setup.sh — starting in background (log: /root/llama-setup.log)"
nohup bash /usr/local/sbin/llama-setup.sh >> /root/llama-setup.log 2>&1 &
echo "pid $!"
echo "Follow with:  tail -f /root/llama-setup.log"
