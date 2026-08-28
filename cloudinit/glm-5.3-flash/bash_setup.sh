#!/usr/bin/env bash
# Writes the setup script and starts it in the background.
# Safe for:  curl -fsSL …/bash_setup.sh | bash
# Ubuntu 22.04/24.04 and AlmaLinux/Rocky/RHEL 8–10.
#
# GLM-5.3-Flash Unsloth UD-IQ3_XXS (~120 GB, glm5next). Target: 2× A100 80GB.
# Stock llama.cpp will not load this — builds unslothai/llama.cpp glm5next/upstream.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root." >&2
  exit 1
fi

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
CTX=65536                    # native max is 1M; 64k is comfortable on 2×80GB IQ3
REASONING=high               # low | high | max  (GLM-5.3-Flash; default in Unsloth is max)
HF_TOKEN=""                  # optional. Or /root/hf-token.txt
################################################################

REPO="unsloth/GLM-5.3-Flash-GGUF"
MODEL_FILE="GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf"
PARTS=(
  "UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf:9429859"
  "UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00002-of-00004.gguf:49261511584"
  "UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00003-of-00004.gguf:49075832480"
  "UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00004-of-00004.gguf:22020797792"
)
MMPROJ_REL="mmproj-F16.gguf"
MMPROJ_SIZE=1128047200
IMAGE=llama-glm5next:local

. /etc/os-release
case "${ID:-}" in
  ubuntu|debian) PKG=apt ;;
  almalinux|rocky|rhel|centos) PKG=dnf ;;
  *) echo "Unsupported OS: ${ID:-unknown} (need Ubuntu/Debian or Alma/Rocky/RHEL)" >&2; exit 1 ;;
esac

if [ "$PKG" = "apt" ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y curl ca-certificates gnupg openssl python3
else
  dnf -y install dnf-plugins-core curl ca-certificates gnupg2 openssl python3
fi

if [ -f /root/llama-api-key.txt ]; then
  API_KEY=$(cat /root/llama-api-key.txt)
elif [ "$API_KEY" = "CHANGE-ME" ]; then
  API_KEY=$(openssl rand -hex 24)
fi
echo "$API_KEY" > /root/llama-api-key.txt
chmod 600 /root/llama-api-key.txt

set +x
if [ -z "$HF_TOKEN" ]; then
  for tokfile in /root/hf-token.txt /root/.cache/huggingface/token /root/.huggingface/token; do
    if [ -f "$tokfile" ]; then
      HF_TOKEN=$(tr -d '[:space:]' < "$tokfile")
      [ -n "$HF_TOKEN" ] && break
    fi
  done
fi
set -x

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

# Spheron: ~96 GB OS disk, ~1.5 TB at /ephemeral.
pick_big_disk() {
  local d avail
  for d in /ephemeral /mnt /data /opt/data; do
    [ -d "$d" ] || continue
    avail=$(df -B1 --output=avail "$d" 2>/dev/null | tail -n1 | tr -d ' ')
    [ -n "$avail" ] && [ "$avail" -gt 200000000000 ] && { echo "$d"; return 0; }
  done
  return 1
}

set_docker_data_root() {
  local root="$1"
  mkdir -p "$root" /etc/docker
  python3 - "$root" <<'PY'
import json, os, sys
root = sys.argv[1]
path = "/etc/docker/daemon.json"
data = {}
if os.path.exists(path) and os.path.getsize(path):
    with open(path) as f:
        data = json.load(f)
data["data-root"] = root
with open(path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY
}

BIG_DISK=$(pick_big_disk || true)
if [ -n "${BIG_DISK:-}" ]; then
  echo "Using ${BIG_DISK} for models + Docker (OS disk is too small for this GGUF)"
  mkdir -p "${BIG_DISK}/models" "${BIG_DISK}/docker"
  if [ -e /models ] && [ ! -L /models ]; then
    mv /models/* "${BIG_DISK}/models/" 2>/dev/null || true
    rm -rf /models
  fi
  ln -sfn "${BIG_DISK}/models" /models
  systemctl stop docker docker.socket 2>/dev/null || true
  if [ -d /var/lib/docker ] && [ "$(ls -A /var/lib/docker 2>/dev/null || true)" ]; then
    echo "Moving /var/lib/docker -> ${BIG_DISK}/docker"
    cp -a /var/lib/docker/. "${BIG_DISK}/docker/"
    rm -rf /var/lib/docker
  fi
  set_docker_data_root "${BIG_DISK}/docker"
fi
mkdir -p /models

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
  if [ -n "${BIG_DISK:-}" ]; then
    set_docker_data_root "${BIG_DISK}/docker"
  fi
  systemctl restart docker
fi

mkdir -p /tmp/llama-docker
cat > /tmp/llama-docker/Dockerfile <<'DF'
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git cmake build-essential curl ca-certificates libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch glm5next/upstream https://github.com/unslothai/llama.cpp.git .
RUN cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build -j"$(nproc)" --config Release --target llama-server
ENTRYPOINT ["/src/build/bin/llama-server"]
DF
docker build -t "$IMAGE" /tmp/llama-docker

docker run --rm --gpus all "$IMAGE" --version || {
  nvidia-ctk runtime configure --runtime=docker
  if [ -n "${BIG_DISK:-}" ]; then
    set_docker_data_root "${BIG_DISK}/docker"
  fi
  systemctl restart docker
  docker run --rm --gpus all "$IMAGE" --version
}

NEED=0
for spec in "${PARTS[@]}" "${MMPROJ_REL}:${MMPROJ_SIZE}"; do
  rel=${spec%%:*}
  base=${rel##*/}
  want=${spec##*:}
  have=0
  [ -f "/models/$base" ] && have=$(stat -c%s "/models/$base")
  if [ "$have" -lt "$want" ]; then
    NEED=$((NEED + want - have))
  fi
done
NEED=$((NEED + 2147483648))
AVAIL=$(df -B1 --output=avail /models | tail -n1 | tr -d ' ')
echo "Download needs ~${NEED} more bytes; /models has ${AVAIL} free ($(df -h /models | tail -n1))"
if [ "$AVAIL" -lt "$NEED" ]; then
  echo "Not enough disk on $(readlink -f /models || echo /models). On Spheron use /ephemeral, not the 96G OS disk." >&2
  df -h
  exit 1
fi

hf_curl() {
  local rc=0
  set +x
  if [ -n "${HF_TOKEN:-}" ]; then
    curl -H "Authorization: Bearer ${HF_TOKEN}" "$@" || rc=$?
  else
    curl "$@" || rc=$?
  fi
  set -x
  return "$rc"
}

dl() { # hf_relative_path expected_bytes
  local rel="$1" want="$2"
  local base="${rel##*/}"
  local f="/models/$base" sz rc=0
  if [ -f "$f" ]; then
    sz=$(stat -c%s "$f")
    if [ "$sz" -eq "$want" ]; then return 0; fi
    if [ "$sz" -lt 1048576 ]; then
      echo "junk $f ($sz bytes) — deleting"
      rm -f "$f"
    else
      echo "resuming $f ($sz / $want bytes)"
    fi
  fi
  hf_curl -fL --retry 10 --retry-delay 5 --retry-connrefused -C - \
    -o "$f" \
    "https://huggingface.co/${REPO}/resolve/main/${rel}?download=true" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "curl failed rc=$rc writing $f" >&2
    df -h
    exit "$rc"
  fi
  [ "$(stat -c%s "$f")" -eq "$want" ] || { echo "SIZE MISMATCH: $f"; df -h; exit 1; }
}

hf_probe="${PARTS[0]%%:*}"
echo "Probing Hugging Face for ${hf_probe}"
set +x
if [ -n "${HF_TOKEN:-}" ]; then
  hf_code=$(curl -sS -o /tmp/hf-probe.body -w "%{http_code}" -L --max-time 30 \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    "https://huggingface.co/${REPO}/resolve/main/${hf_probe}?download=true" \
    -r 0-0 || true)
else
  hf_code=$(curl -sS -o /tmp/hf-probe.body -w "%{http_code}" -L --max-time 30 \
    "https://huggingface.co/${REPO}/resolve/main/${hf_probe}?download=true" \
    -r 0-0 || true)
fi
set -x
if [ "$hf_code" != "200" ] && [ "$hf_code" != "206" ]; then
  echo "Hugging Face refused the download (HTTP ${hf_code})." >&2
  echo "If this is rate-limited, put a READ token in /root/hf-token.txt and re-run." >&2
  echo "  https://huggingface.co/${REPO}" >&2
  sed -n '1,20p' /tmp/hf-probe.body >&2 || true
  exit 1
fi
rm -f /tmp/hf-probe.body
for spec in "${PARTS[@]}"; do
  dl "${spec%%:*}" "${spec##*:}"
done
dl "$MMPROJ_REL" "$MMPROJ_SIZE"

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
if [ "$TOTAL_VRAM_MB" -lt 140000 ]; then
  echo "IQ3_XXS wants ~128–150 GB total memory. This box has ${TOTAL_VRAM_MB} MiB VRAM — 2× A100 80GB is the target." >&2
fi

CHAT_KWARGS=( --chat-template-kwargs "{\"reasoning_effort\":\"${REASONING}\"}" )

docker rm -f llama 2>/dev/null || true
docker run -d --name llama --restart unless-stopped --gpus all \
  --shm-size 16g \
  -p "${PORT}:8080" -v /models:/models \
  "$IMAGE" \
  -m "/models/${MODEL_FILE}" \
  --mmproj "/models/${MMPROJ_REL##*/}" \
  --host 0.0.0.0 --port 8080 \
  --api-key "$API_KEY" \
  -ngl 999 -c "$CTX" --jinja \
  --alias glm-5.3-flash \
  --flash-attn on -np 1 \
  --cache-ram -1 \
  -b 2048 -ub 2048 \
  --threads "$THREADS" --threads-http "$HTTP_THREADS" \
  --temp 1.0 --top-p 0.95 \
  "${CHAT_KWARGS[@]}"

if command -v firewall-cmd >/dev/null && systemctl is-active --quiet firewalld; then
  firewall-cmd --permanent --add-port="${PORT}/tcp" || true
  firewall-cmd --reload || true
fi

systemctl disable llama-setup-continue.service 2>/dev/null || true
rm -f /etc/systemd/system/llama-setup-continue.service
systemctl daemon-reload 2>/dev/null || true

echo "DONE. API key: $API_KEY"
echo "Model: $MODEL_FILE  ctx=$CTX  gpus=$NGPU  reasoning=$REASONING"
SETUP_EOF

chmod 755 /usr/local/sbin/llama-setup.sh
ln -sfn /usr/local/sbin/llama-setup.sh /root/setup.sh
restorecon -v /usr/local/sbin/llama-setup.sh 2>/dev/null \
  || chcon -t bin_t /usr/local/sbin/llama-setup.sh 2>/dev/null || true
echo "Wrote /usr/local/sbin/llama-setup.sh — starting in background (log: /root/llama-setup.log)"
nohup bash /usr/local/sbin/llama-setup.sh >> /root/llama-setup.log 2>&1 &
echo "pid $!"
echo "Follow with:  tail -f /root/llama-setup.log"
