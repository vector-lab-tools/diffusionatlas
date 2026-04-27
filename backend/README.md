# Diffusion Atlas — Local Backend

FastAPI service that wraps the [diffusers](https://huggingface.co/docs/diffusers) library so Atlas operations needing per-step latents (Denoise Trajectory) or true latent-space sampling can run on your own hardware. Optional — hosted operations work without it.

## Setup

Python 3.11 or later, with a GPU (CUDA) or Apple Silicon (MPS). CPU works but will be very slow.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first model load downloads weights from Hugging Face. For gated models (SD3, FLUX-dev) accept the licence on HF and run `huggingface-cli login` once.

## Run

```bash
uvicorn main:app --reload --port 8000
```

Then in Diffusion Atlas (http://localhost:3000), open Settings → switch Backend to **Local FastAPI**, confirm the URL is `http://localhost:8000`, and run any operation. The dispatcher will call this service.

## Endpoints

- `GET /health` — device, dtype, torch version, current model, total memory (best-effort).
- `POST /generate` — single image generation. Body: `{ modelId, prompt, seed, steps, cfg, width, height, negativePrompt? }`. Returns `{ images: [data url], meta }`.
- `POST /trajectory` — *coming in v0.2.1* — NDJSON stream of per-step latents for Denoise Trajectory.

## Capabilities by device

| Device | Default dtype | Notes |
|---|---|---|
| CUDA | fp16 | Best path; FLUX-dev and SD3 viable on 24 GB+. |
| MPS (Apple Silicon) | fp16 | Attention slicing + VAE tiling enabled to fit SDXL on 24 GB unified. SD 1.5 is the safe dev default. |
| CPU | fp32 | Smoke tests only. |

`/health` reports which device was detected so the frontend can hint at compatible models.

## Models

Models are pulled by Hugging Face id at first request. Suggested registry (subject to user override):

- `runwayml/stable-diffusion-v1-5` — ~4 GB, the safe default
- `stabilityai/stable-diffusion-xl-base-1.0` — ~10 GB, needs slicing on 24 GB MPS
- `stabilityai/stable-diffusion-3-medium-diffusers` — license-gated
- `black-forest-labs/FLUX.1-schnell` — license-gated, distilled (4 steps)
- `black-forest-labs/FLUX.1-dev` — license-gated, ~24 GB

The session keeps one pipeline in memory at a time. Switching `modelId` between requests triggers a swap.

## Licence

MIT, same as the parent project.
