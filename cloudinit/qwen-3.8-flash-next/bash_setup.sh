#!/usr/bin/env bash
# Writes the setup script and starts it in the background.
# Safe for:  curl -fsSL …/bash_setup.sh | bash
# Ubuntu 22.04/24.04 and AlmaLinux/Rocky/RHEL 8–10.
#
# Qwen3.8-Flash-Next Uncensored GGUF (qwen4exp). Needs llama.cpp from 2026-08-27+
# (PR #27742). Default quant is Q5_K_M (~125 GiB). There is no Q6 — the PLE
# tensor would exceed HF's 50 GB file cap. Target box: 4× A100 80GB.
# Native window is 262144; default is YaRN ×4 to 1048576 (Qwen's 1M max).
# 2×80GB keeps native 262k (hybrid KV is ~6 GiB at 262k — only 12 of 48 layers
# grow a cache; ~24 GiB at 1M, which is the 4× box).
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
CTX=1048576                  # YaRN ×4 = 1M (Qwen max). Native 262144. 524288 = ×2
YARN=1                       # 0 = native window only (no RoPE scaling)
CTX_FORCE=0                  # 1 = keep CTX/YARN even on 2×80GB
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

if [ -z "$HF_TOKEN" ]; then
  set +x
  for tokfile in /root/hf-token.txt /root/.cache/huggingface/token /root/.huggingface/token; do
    if [ -f "$tokfile" ]; then
      HF_TOKEN=$(tr -d '[:space:]' < "$tokfile")
      [ -n "$HF_TOKEN" ] && break
    fi
  done
  set -x
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

# Spheron (and similar) put a tiny OS disk on / and the real NVMe at /ephemeral.
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

if [ "$BUILD_LLAMA" = "1" ]; then
  IMAGE=llama-qwen4exp:local
  mkdir -p /tmp/llama-docker
  # llama.cpp #27835/#27871: QSA indexer RMS-norm puts n_blocks on CUDA gridDim.y
  # (limit 65535), so n_kv>=262144 aborts with "invalid argument". Swap so the
  # large count lands on gridDim.x. Idempotent if master already has the fix.
  cat > /tmp/llama-docker/patch_qwen4exp.py <<'PY'
from pathlib import Path
import re
import sys

p = Path("src/models/qwen4exp.cpp")
t = p.read_text()
if "n_blocks*n_stream overflows at n_kv" in t or "normalize before flattening" in t:
    print("qwen4exp indexer RMS-norm already patched")
    sys.exit(0)
pat = re.compile(
    r"([ \t]*)// rope wants \[n_dims, n_head, n_tokens\]: lay every stream's blocks flat, split after\.\n"
    r"([ \t]*)pooled = ggml_reshape_3d\(ctx0, pooled, idx_dim, 1, n_blocks\*n_stream\);\n"
    r"([ \t]*)pooled = build_norm\(pooled, model\.layers\[il\]\.index_k_norm, nullptr, LLM_NORM_RMS, il\);"
)
m = pat.search(t)
if not m:
    sys.exit("qwen4exp.cpp indexer RMS-norm pattern not found")
ind = m.group(1)
new = (
    f"{ind}// normalize before flattening: rms_norm puts ne[1] on gridDim.x but ne[2]\n"
    f"{ind}// on gridDim.y (limit 65535). n_blocks*n_stream overflows at n_kv>=262144 (#27835).\n"
    f"{ind}pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);\n"
    f"{ind}// rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.\n"
    f"{ind}pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_blocks*n_stream);"
)
p.write_text(t[: m.start()] + new + t[m.end() :])
print("patched qwen4exp indexer RMS-norm order")
PY
  cat > /tmp/llama-docker/Dockerfile <<'DF'
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git cmake build-essential curl ca-certificates libcurl4-openssl-dev python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git .
COPY patch_qwen4exp.py /tmp/patch_qwen4exp.py
RUN python3 /tmp/patch_qwen4exp.py
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
  if [ -n "${BIG_DISK:-}" ]; then
    set_docker_data_root "${BIG_DISK}/docker"
  fi
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
NEED=0
for spec in "${PARTS[@]}" "$MMPROJ_FILE:$MMPROJ_SIZE"; do
  f="/models/${spec%%:*}"
  want=${spec##*:}
  have=0
  [ -f "$f" ] && have=$(stat -c%s "$f")
  if [ "$have" -lt "$want" ]; then
    NEED=$((NEED + want - have))
  fi
done
NEED=$((NEED + 2147483648))
AVAIL=$(df -B1 --output=avail /models | tail -n1 | tr -d ' ')
echo "Download needs ~${NEED} more bytes; /models has ${AVAIL} free ($(df -h /models | tail -n1))"
if [ "$AVAIL" -lt "$NEED" ]; then
  echo "Not enough disk on $(readlink -f /models || echo /models). On Spheron use /ephemeral (1.5T), not the 96G OS disk." >&2
  df -h
  exit 1
fi

dl() { # filename expected_bytes
  local f="/models/$1" sz rc=0
  if [ -f "$f" ]; then
    sz=$(stat -c%s "$f")
    if [ "$sz" -eq "$2" ]; then return 0; fi
    if [ "$sz" -lt 1048576 ]; then
      echo "junk $f ($sz bytes) — deleting (failed HTML/401 stub)"
      rm -f "$f"
    else
      echo "resuming $f ($sz / $2 bytes)"
    fi
  fi
  set +x
  curl -fL --retry 10 --retry-delay 5 --retry-connrefused -C - \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    -o "$f" \
    "https://huggingface.co/${REPO}/resolve/main/$1?download=true" || rc=$?
  set -x
  if [ "$rc" -ne 0 ]; then
    echo "curl failed rc=$rc writing $f" >&2
    df -h
    exit "$rc"
  fi
  [ "$(stat -c%s "$f")" -eq "$2" ] || { echo "SIZE MISMATCH: $f"; df -h; exit 1; }
}

hf_probe="${PARTS[0]%%:*}"
echo "Probing Hugging Face auth for ${hf_probe}"
set +x
hf_code=$(curl -sS -o /tmp/hf-probe.body -w "%{http_code}" -L --max-time 30 \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  "https://huggingface.co/${REPO}/resolve/main/${hf_probe}?download=true" \
  -r 0-0 || true)
set -x
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

NATIVE_CTX=262144
YARN_CTX=1048576
# Hybrid QSA: only 12/48 layers grow KV (~6 GiB f16 at 262k, ~24 GiB at 1M).
# 2×80GB holds Q5_K_M + native 262k. YaRN to 1M is the 4× box.
if [ "$CTX_FORCE" != "1" ] && [ "$TOTAL_VRAM_MB" -lt 280000 ]; then
  if [ "$YARN" = "1" ] || [ "$CTX" -gt "$NATIVE_CTX" ]; then
    echo "VRAM ${TOTAL_VRAM_MB} MiB is below 4×80GB; native ${NATIVE_CTX} (no YaRN)"
    echo "Rent 4× A100 for ${YARN_CTX}, or set CTX_FORCE=1 to try anyway."
    YARN=0
    CTX=$NATIVE_CTX
  fi
fi
if [ "$YARN" = "1" ] && [ "$CTX" -le "$NATIVE_CTX" ]; then
  CTX=$YARN_CTX
fi
YARN_ARGS=()
if [ "$YARN" = "1" ]; then
  ROPE_SCALE=$(awk -v c="$CTX" -v n="$NATIVE_CTX" 'BEGIN{printf "%.8g", c/n}')
  # llama-server otherwise caps slots at n_ctx_train (262144).
  YARN_ARGS=(
    --rope-scaling yarn --rope-scale "$ROPE_SCALE" --yarn-orig-ctx "$NATIVE_CTX"
    --override-kv "qwen4exp.context_length=int:${CTX}"
  )
  echo "YaRN: CTX=$CTX rope-scale=$ROPE_SCALE orig=$NATIVE_CTX"
elif [ "$NGPU" -ge 4 ]; then
  echo "4+ GPUs: CTX=$CTX (native Flash-Next window, YARN=0)"
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
  "${YARN_ARGS[@]}" \
  "${PLE_ARGS[@]}"

if command -v firewall-cmd >/dev/null && systemctl is-active --quiet firewalld; then
  firewall-cmd --permanent --add-port="${PORT}/tcp" || true
  firewall-cmd --reload || true
fi

systemctl disable llama-setup-continue.service 2>/dev/null || true
rm -f /etc/systemd/system/llama-setup-continue.service
systemctl daemon-reload 2>/dev/null || true

echo "DONE. API key: $API_KEY"
echo "Model: $MODEL_FILE  ctx=$CTX  yarn=$YARN  gpus=$NGPU"
SETUP_EOF

chmod 755 /usr/local/sbin/llama-setup.sh
ln -sfn /usr/local/sbin/llama-setup.sh /root/setup.sh
restorecon -v /usr/local/sbin/llama-setup.sh 2>/dev/null \
  || chcon -t bin_t /usr/local/sbin/llama-setup.sh 2>/dev/null || true
echo "Wrote /usr/local/sbin/llama-setup.sh — starting in background (log: /root/llama-setup.log)"
nohup bash /usr/local/sbin/llama-setup.sh >> /root/llama-setup.log 2>&1 &
echo "pid $!"
echo "Follow with:  tail -f /root/llama-setup.log"
