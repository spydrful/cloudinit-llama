cat > /root/setup.sh <<'EOF'
#!/usr/bin/env bash
set -euxo pipefail

##### EDIT ME ##################################################
API_KEY="CHANGE-ME"          # if left as-is, random one -> /root/llama-api-key.txt
PORT=8080
CTX=262144                   # model native max; Q8_0 + vision on 80GB A100
################################################################

REPO="JonathanColetti/Qwen3.8-27B-Uncensored-GGUF"
MODEL_FILE="Qwen3.8-27B-Uncensored-Q8_0.gguf"
MODEL_SIZE=29047084448
# fallback if MTP quant misbehaves (no MTP tensors; slightly smaller):
# MODEL_FILE="Qwen3.8-27B-Uncensored-noMTP-Q8_0.gguf"
# MODEL_SIZE=28595763680
MMPROJ_FILE="Qwen3.8-27B-Uncensored-vision-f16.gguf"
MMPROJ_SIZE=927606912

if [ "$API_KEY" = "CHANGE-ME" ]; then
  API_KEY=$(openssl rand -hex 24)
fi
echo "$API_KEY" > /root/llama-api-key.txt
chmod 600 /root/llama-api-key.txt

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y curl ca-certificates gnupg openssl

nvidia-smi
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh

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

docker rm -f llama 2>/dev/null || true
docker run -d --name llama --restart unless-stopped --gpus all \
  -p "${PORT}:8080" -v /models:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m "/models/${MODEL_FILE}" \
  --mmproj "/models/${MMPROJ_FILE}" \
  --host 0.0.0.0 --port 8080 \
  --api-key "$API_KEY" \
  -ngl 999 -c "$CTX" --jinja \
  --alias qwen3.8-27b-uncensored

echo "DONE. API key: $API_KEY"
