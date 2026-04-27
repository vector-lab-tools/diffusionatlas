> Part of the [Vector Lab](https://github.com/vector-lab-tools) — vector methods for vector theory.
> [Overview and map](https://vector-lab-tools.github.io) · [Org profile](https://github.com/vector-lab-tools)
>
> **Tier:** comparative model tool. **Object:** generative trajectories of diffusion models.
>
> **Sibling instruments:** [Vectorscope](https://github.com/vector-lab-tools/vectorscope) · [Manifold Atlas](https://github.com/vector-lab-tools/manifold-atlas) · [Manifoldscope](https://github.com/vector-lab-tools/manifoldscope) · [Theoryscope](https://github.com/vector-lab-tools/theoryscope) · [LLMbench](https://github.com/vector-lab-tools/LLMbench)

# Diffusion Atlas

**Manifold geometry and benchmarking for diffusion models.**

**Author:** David M. Berry
**Institution:** University of Sussex
**Version:** 0.3.2
**Date:** 28 April 2026
**Licence:** MIT

Diffusion Atlas is a vector-native research tool for studying how diffusion models generate images geometrically. Where [Manifold Atlas](https://github.com/vector-lab-tools/manifold-atlas) reads the static manifold of a frozen embedding model, Diffusion Atlas reads the generative trajectory of a denoising model: the path through latent space that produces an image. It unifies an **Atlas** view (interpretability and geometry) with a **Bench** view (scored compositional evaluation) in a single instrument.

The tool extends [Vector Theory](https://stunlaw.blogspot.com/2026/02/vector-theory.html) from text-embedding manifolds to generative diffusion. The asymmetry between the two regimes is itself a finding: the manifold framing migrates more cleanly to diffusion than to autoregressive text, because in diffusion the latent space is genuinely geometric rather than inferred from token logits. Position, orientation, and proximity are not metaphors here; they are the substrate on which generation runs.

## Scholarly Context

Diffusion Atlas emerges from three converging research programmes.

**Vector theory.** Berry (2026) *Vector Theory* argues that the vectorial turn introduces a new computational regime in which definition is replaced by position, truth by orientation, argument by interpolation, and contradiction by cosine proximity. Diffusion models are this regime in its most explicit form: the entire generative process is a trajectory through a learned latent geometry. Diffusion Atlas operationalises this view with Atlas operations that read the geometry directly.

**Compositionality and the limits of generation.** Diffusion models excel at texture and aesthetic coherence but fail in characteristic ways on compositional tasks: object counts, attribute binding, spatial relations, negation. The GenEval benchmark and successors have made these failures legible. The Bench view in Diffusion Atlas operationalises compositional evaluation alongside Atlas-side geometry so that the cost of a model's geometric decisions is visible in the same session.

**Comparative, multi-backend method.** No single backend gives an honest picture of a diffusion model. Hosted APIs hide intermediate latents; local backends require GPU access. Diffusion Atlas is built to run the same operation across both, treating capability differences as findings rather than friction.

## Atlas vs Bench: A Primer

Diffusion Atlas distinguishes two analytical surfaces that most diffusion tools keep apart.

**Atlas operations** treat diffusion as a vector process. They read the geometry of latent space, the shape of denoising trajectories, and the structure of neighbourhoods around a chosen point. They are interpretive: they ask where in the latent space the model decides, where the manifold thickens, where it thins, and where small perturbations produce categorical jumps in the output.

**Bench operations** treat diffusion as a system to be scored. They run controlled task packs (single object, two objects, counting, colour, spatial relations) against the model and report per-category accuracy. They are evaluative: they ask what the model's geometric decisions cost in compositional fidelity.

The two views run on the same generated images. A finding in Bench (the model fails attribute binding for red-and-blue object pairs) can be cross-checked in Atlas (does the latent neighbourhood for that prompt show two distinct basins, or one collapsed mode?). This is the methodological reason for a single app rather than two.

## Operations at a Glance (v0.2)

| Operation | View | Core question | Backend |
|---|---|---|---|
| Denoise Trajectory | Atlas | What path does the model take through latent space? | **Local** (NDJSON stream) |
| Guidance Sweep | Atlas | How does CFG bend the trajectory? Where does mode collapse? | Hosted or Local |
| Latent Neighbourhood | Atlas | What does the local manifold look like around an anchor? | Hosted or Local |
| Compositional Bench | Bench | How well does the model bind, count, and place? | Hosted or Local |
| Library | — | What did we run, and what did it produce? | Local (IndexedDB) |

## Features

### Denoise Trajectory
Trace the iterative denoising path through latent space. The local backend streams per-step latents over NDJSON, with a thumbnail decoded through the VAE every Nth step (configurable via the **Preview every** field). The client projects the latents to 3D via PCA and renders the path in Three.js with start (gold) and end (burgundy) markers; thumbnails appear as billboard sprites above their corresponding step markers, so you can see the image taking shape along the curve. Local backend required: hosted providers do not expose intermediate latents.

### Guidance Sweep
Generate the same prompt and seed across a list of CFG values (default `1, 2.5, 4, 7.5, 12`). The image grid is keyed by CFG with per-cell status while the run is in flight. The sweep is sequential per lane rather than `Promise.all` so a single failure doesn't tank the run, and rate-limited responses (Replicate's free tier triggers them quickly) are honoured: each cell shows a live `Retrying in Ns` countdown then resumes. A drift curve above each grid plots normalised perceptual-hash distance from the baseline (CFG nearest 7.5), so the controllability surface — and where mode collapse begins — is visible at a glance.

**Cross-backend comparison** (v0.3.0): tick *Compare with a second provider* to run a parallel sweep against any other configured provider (Replicate ↔ Fal, hosted ↔ Local, etc.). Both lanes share the same prompt, seed, steps, and CFG list and run in parallel — independent rate limits, independent retry loops. Side-by-side grids and drift curves make geometry that is structural (consistent across backends) distinguishable from geometry that is contingent (specific to one provider's training).

### Latent Neighbourhood
Sample k images around an anchor seed at a configurable radius. Deterministic seed offsets so the run is reproducible. Hosted mode samples by varying the seed (each seed maps to a different starting latent); true Gaussian perturbation of the initial latent at a chosen sigma is queued for the local backend.

**Cross-backend comparison** (v0.3.1): tick *Compare with a second provider* to run the same anchor + seed offsets against a second provider in parallel. Two thumbnail grids stack with their provider labels, so you can see how a 24 GB local SD 1.5 and a hosted flux-schnell organise the local manifold around the same prompt. Where the two neighbourhoods cluster differently is where the geometry is contingent on the model rather than structural to diffusion.

### Compositional Bench
GenEval-lite task pack: 4 categories (single object, two objects, counting, colour binding) × 3 prompts each = 12 tasks. Generate the pack at a fixed seed; mark each result pass or fail with the live per-category scoring panel, **or** click *Auto-score (CLIP)* to score every image against its prompt with `openai/clip-vit-base-patch32` on the local backend and set verdicts from cosine similarity at a configurable threshold (0.25 is the conventional cutoff). Each card shows the numeric score; verdicts can be overridden manually at any time.

### Library
All saved runs across Sweep, Neighbourhood, Bench, and Trajectory, grouped by kind, newest first. Stored entirely in the browser's IndexedDB; nothing leaves the machine. Per-run delete and a Clear all action.

## Supported Diffusion Backends

Diffusion Atlas supports two classes of backend, selected per-session in Settings.

### Hosted (default, lowest setup friction)

Hosted providers return final images via API. They are the right choice for Guidance Sweep, Latent Neighbourhood (seed-jitter mode), and Compositional Bench. They cannot serve Denoise Trajectory because they do not expose intermediate latents.

| Provider | Status | Notes | Sign up |
|----------|--------|-------|---------|
| Replicate | wired | Broad model selection (SDXL, SD3, FLUX, custom). Aggressive rate limits on the free tier. | [replicate.com](https://replicate.com/) |
| Fal | wired | An order of magnitude more permissive on rate limits than Replicate. Best choice for sweep- and bench-heavy work. | [fal.ai](https://fal.ai/) |
| Together | planned | OpenAI-compatible inference, image and chat | [together.ai](https://www.together.ai/) |
| Stability AI | planned | Stability's own SD3, SDXL, and Stable Image series | [platform.stability.ai](https://platform.stability.ai/) |

When you switch providers in Settings, the model id auto-updates to a sensible starting point for that provider (Replicate uses `owner/model` ids; Fal uses `fal-ai/<model>`). Override afterwards as needed.

### Local (full latent access)

The local backend is a small FastAPI service that wraps the [diffusers](https://huggingface.co/docs/diffusers) library. It runs on your own hardware (CUDA or Apple Silicon MPS) and is the only way to run Denoise Trajectory or true latent-space neighbourhood sampling. No API key is needed; no data leaves your machine. Suitable models include Stable Diffusion 1.5, SDXL, SD3, and FLUX-schnell, depending on memory.

To use the local backend, install the FastAPI service in `backend/` (see Getting Started) and select **Local** in Settings.

## Design Rationale

**Why both Atlas and Bench in one app?** In diffusion the manifold *is* the benchmark. Controllability, compositional generalisation, and mode coverage are all geometric properties of the latent space. Splitting them into separate tools forces you to compute the same geometry twice and pretend the scores are independent. Diffusion Atlas treats Bench as a derived view of the Atlas substrate so a finding in one is checkable in the other.

**Why hosted plus local?** Hosted is the path of least resistance and serves most of the Bench view and the cheaper Atlas operations. Local is the only way to study the trajectory itself, because hosted providers do not expose intermediate latents. Treating them as peer providers, with capability mismatches surfaced as typed errors, lets the same operation degrade gracefully across backends.

**Why cache latents and images in IndexedDB?** A single trajectory at 30 steps and 1024x1024 resolution can take 5 to 30 seconds. Identical queries should never cost twice. Latents (Float32Array) and images (Blob) are cached deterministically by full generation parameters. Image blobs use LRU eviction with a configurable cap (default 500 MB).

**Why a browser-only frontend?** The instrument is for research. Running entirely in the browser, with API keys stored client-side and the local backend on localhost, keeps the deployment surface minimal and the data on the researcher's own machine. No tracking, no telemetry.

**Why editable `models/*.md` files?** The pace of model releases outruns any sensible rebuild cadence. Keeping the model registry in markdown lets researchers add a new model as soon as it appears, without touching compiled artefacts.

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) 18 or later
- For Atlas operations beyond Guidance Sweep: a GPU (CUDA) or Apple Silicon (MPS) with at least 12 GB of available memory
- For hosted operations: an API key from at least one of Replicate, Fal, Together, or Stability AI

### Install and Run (frontend)

```bash
git clone https://github.com/vector-lab-tools/diffusionatlas.git
cd diffusion-atlas
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Install and Run (local FastAPI backend, optional)

The local backend is required for Denoise Trajectory and true latent-space neighbourhood sampling. It is optional otherwise.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The backend listens on `http://localhost:8000`. In Diffusion Atlas Settings, switch the backend to **Local** and confirm the URL.

### Configure Providers

1. Click **Settings** (top right)
2. Choose **Hosted** or **Local** as the backend
3. For hosted: select a provider and paste your API key
4. For local: confirm the FastAPI URL and select a model
5. Close settings and start generating

### Adding or Removing Models

Model lists are defined in markdown under `public/models/`, one file per provider (`replicate.md`, `fal.md`, `together.md`, `stability.md`, `local.md`). Add a model with one line:

```
model-id | Display Name | notes
```

Save the file and reload the app. Lines starting with `#` are comments.

## Architecture

```
src/
  app/
    api/diffuse/         # Hosted + local image generation dispatcher
    api/trajectory/      # Local-only NDJSON stream of per-step latents
    api/bench/           # Scored task runs
    api/keys/            # Server-side API key proxy (optional)
    api/local-diffuse/   # Proxy to FastAPI backend
  components/
    operations/          # Denoise Trajectory, Guidance Sweep,
                         # Latent Neighbourhood, Compositional Bench
    viz/                 # TrajectoryThree, GuidanceGridPlot,
                         # NeighbourhoodScatter, BenchLeaderboard
    layout/              # Header, TabNav, StatusBar, SettingsPanel,
                         # AboutModal, HelpDropdown
    shared/              # ResultCard, DeepDive, ImageGalleryGrid,
                         # PromptChips, ErrorDisplay, OperationStub
  context/               # DiffusionSettingsContext, LatentCacheContext,
                         # ImageBlobCacheContext
  lib/
    providers/           # Replicate, Fal, Together, Stability, Local
                         # plus dispatcher and shared types
    operations/          # Pure compute for each operation
    geometry/            # PCA, UMAP, latent distance utilities
    bench/               # GenEval-lite tasks and scoring
    cache/               # IndexedDB wrappers
    export/              # PDF and CSV
  types/                 # Shared TypeScript types

backend/                 # Local FastAPI service (optional)
  main.py                # FastAPI app
  config/                # CORS, settings
  models/                # diffusers pipeline session
  operations/            # denoise_trajectory, guidance_sweep,
                         # latent_neighbourhood, generate
  geometry/              # latent reduce / IO
```

Latent vectors and image blobs are cached in IndexedDB. Settings persist in localStorage. No server-side database, no authentication.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 16 (App Router), React 19 |
| Language | TypeScript 5 (strict) |
| Styling | Tailwind CSS 3, CCS-WB editorial design system |
| Visualisation | Plotly.js (GL3D), Three.js (@react-three/fiber) |
| Dimensionality reduction | umap-js (browser-side), custom PCA |
| Caching | IndexedDB via idb |
| Validation | Zod |
| Local backend | FastAPI, diffusers, PyTorch |

## Theoretical Context

Diffusion Atlas is a research instrument for the vector theory programme developed by David M. Berry. The vectorial turn introduces a new computational regime in which meaning is encoded as position and inference is performed as movement through a learned manifold. Diffusion models are this regime in its most explicit form: the entire generative process is a trajectory through latent space, and the final image is a projection of where that trajectory ended.

The Atlas operations test specific claims of the framework. Denoise Trajectory makes the path of generation visible as a curve. Guidance Sweep tests how aggressively the conditional vector pushes the trajectory off its unconditional course. Latent Neighbourhood tests the local geometry of the manifold around any chosen point. Compositional Bench, on the Bench side, measures the cost of these geometric decisions in tasks that require categorical compositionality (binding, counting, placement) which the manifold tends to handle as smooth interpolation rather than discrete combination.

## Roadmap

- [x] App shell with Atlas / Bench / Library tabs (v0.1.0)
- [x] Vector Lab branding and theme-aware tool icon (v0.1.1)
- [x] Provider abstraction with Replicate hosted provider (v0.1.2 – 0.1.7)
- [x] Guidance Sweep with image grid and rate-limit retry (v0.1.8 – 0.1.9)
- [x] Latent Neighbourhood with seed jitter (v0.1.10)
- [x] Compositional Bench (GenEval-lite) with manual scoring (v0.1.11)
- [x] Library browse for saved runs (v0.1.12)
- [x] Local FastAPI backend skeleton with /generate (v0.2.0)
- [x] **Denoise Trajectory** with NDJSON streaming and 3D PCA path (v0.2.1)
- [x] Per-step preview thumbnails along the trajectory + drift curve in Guidance Sweep (v0.2.2)
- [x] CLIP-based auto-scoring for Compositional Bench (v0.2.3)
- [x] Fal.ai hosted provider + UMAP toggle on the trajectory projection (v0.2.4)
- [x] Cross-backend agreement view in Guidance Sweep (v0.3.0) — same prompt + seed, two providers in parallel, side-by-side grids and drift curves
- [x] Cross-backend comparison extended to Latent Neighbourhood (v0.3.1)
- [x] Cross-backend comparison extended to Compositional Bench + Deep Dive panels with CSV/PDF/JSON export across all operations (v0.3.2)
- [ ] Clippy / Hackerman easter eggs with diffusion-flavoured quips
- [ ] Object-detection-based bench scoring (proper GenEval rather than CLIP cosine)
- [ ] UMAP option for trajectory and neighbourhood
- [ ] PDF export across operations
- [ ] Fal.ai / Together / Stability hosted providers
- [ ] Attention-map and cross-attention visualisation
- [ ] h-space steering

## Related Work

- Berry, D. M. (2026) 'Vector Theory', *Stunlaw*. Available at: https://stunlaw.blogspot.com/2026/02/vector-theory.html
- Berry, D. M. (2026) 'What is Vector Space?', *Stunlaw*. Available at: https://stunlaw.blogspot.com/2026/03/what-is-vector-space.html
- Berry, D. M. (2026) *Artificial Intelligence and Critical Theory*. MUP.
- Ho, J. and Salimans, T. (2022) 'Classifier-Free Diffusion Guidance'. Available at: https://arxiv.org/abs/2207.12598
- Ghosh, D. et al. (2023) 'GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment'. Available at: https://arxiv.org/abs/2310.11513
- Rombach, R. et al. (2022) 'High-Resolution Image Synthesis with Latent Diffusion Models'. *CVPR*. Available at: https://arxiv.org/abs/2112.10752
- Park, Y.-H. et al. (2023) 'Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry'. Available at: https://arxiv.org/abs/2307.12868

## Acknowledgements

Concept and Design by David M. Berry, implemented with Claude Code. Design system adapted from the [CCS Workbench](https://github.com/dmberry/ccs-wb).

## Licence

MIT
