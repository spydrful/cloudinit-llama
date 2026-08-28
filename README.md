# cloudinit-llama

Scripts for standing up **AI cloud servers** on rented GPU instances — LLM
inference and video generation — with one paste into a cloud-init / bash field.

Each folder is a self-contained setup (script + its own README). They target
NVIDIA hosts (Ubuntu 22.04/24.04; bash installers also cover Alma/RHEL where
noted). 27B Qwen setups fit **1× 80 GB**. GLM-5.3-Flash IQ3 needs **2× 80 GB**.

## What's inside

| Folder | What it deploys | GPU |
|---|---|---|
| [`cloudinit/qwen-3.6/`](cloudinit/qwen-3.6/) | `llama.cpp` OpenAI-compatible LLM server — Qwen3.6-27B Fable-Fusion GGUF (Q8_0) with vision, bearer-token auth. Cloud-init **and** plain-bash forms. | 80 GB |
| [`cloudinit/qwen-3.8/`](cloudinit/qwen-3.8/) | `llama.cpp` OpenAI-compatible LLM server — [Qwen3.8-27B Uncensored](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF) GGUF (Q8_0) with vision, bearer-token auth. Cloud-init **and** plain-bash forms. | 80 GB |
| [`cloudinit/glm-5.3-flash/`](cloudinit/glm-5.3-flash/) | `llama.cpp` OpenAI-compatible LLM server — [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) Unsloth **UD-IQ3_XXS** (~120 GB, `glm5next`). | **2× 80 GB** |
| [`comfyui/A100/`](comfyui/A100/) | ComfyUI + **MiniMax H3** open-weights video model (text/image/reference → video, native audio). | A100 80 GB |

## Shared design

- **Docker + NVIDIA runtime**, verified visible in-container before any large
  download.
- **No secrets in the repo.** Tokens default to `CHANGE-ME` → a random value
  generated on the box (or no auth where the app has none — see per-folder
  security notes).
- **Single-stream, size-verified Hugging Face downloads.** Resumable, and
  immune to the Xet-CDN 403 that breaks multi-connection downloaders.
- **Re-runnable.** Completed files are skipped; interrupted downloads resume.

## Security

Ports face the internet — open them deliberately in the provider firewall.

- The **llama server** ships bearer-token auth; keep the token strong.
- **ComfyUI has no authentication and can run code** — it binds to localhost;
  reach it over an SSH tunnel (see [`comfyui/A100/`](comfyui/A100/)).

Nothing here is a hardened multi-tenant deployment — these are single-user setups
for your own instances.
