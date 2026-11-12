"use client";

import { useState } from "react";
import { HelpCircle, X } from "lucide-react";
import { useEscape } from "@/lib/useEscape";

interface HelpEntry {
  title: string;
  content: string;
  link?: { label: string; url: string };
}

interface HelpPane {
  id: string;
  label: string;
  entries: HelpEntry[];
}

const DOCS_URL = "https://vector-lab-tools.github.io/diffusionatlas/";

const HELP_PANES: HelpPane[] = [
  {
    id: "start",
    label: "Getting started",
    entries: [
      {
        title: "Set up a backend",
        content: "Open Settings (top right). Choose Hosted to generate through Replicate or Fal: pick the provider and paste its API key. Keys are stored in this browser only; Save to .env.local keeps them across dev-server restarts. Choose Local FastAPI to run the diffusers backend on your own GPU or Apple Silicon machine; it listens on http://localhost:8000 and is the only route to Denoise Trajectory. Settings changes apply when you press Save.",
        link: { label: "Step-by-step API key setup", url: `${DOCS_URL}#api-keys` },
      },
      {
        title: "Choose a model",
        content: "The Model field in Settings offers suggestions for the selected provider, but it is free text: any model id from the provider's catalogue can be typed in and is used as entered. Hosted model line-ups change; if a suggestion is withdrawn, type the current id from the provider's model page (Replicate: owner/model; Fal: fal-ai/<model>; Local: a Hugging Face id). Reset to default restores the provider's starting model.",
      },
      {
        title: "Where the operations are",
        content: "The Atlas tab holds Denoise Trajectory (local backend only), Guidance Sweep and Latent Neighbourhood. The Bench tab holds Compositional Bench. The Library tab lists every saved run. Each operation has its own pane in this guide.",
      },
    ],
  },
  {
    id: "concepts",
    label: "Concepts",
    entries: [
      {
        title: "What is a Diffusion Model?",
        content: "A diffusion model generates an image by reversing a noise process. It starts with pure noise and, over a fixed number of steps, denoises that noise into a coherent image. At each step the model predicts what to remove, conditioned on a text prompt. The trajectory through this denoising process is the object of study for Diffusion Atlas: the same final image can be reached by very different paths, and what the model 'decides' along the way is often more revealing than the picture it ends on.",
      },
      {
        title: "Atlas vs Bench",
        content: "Diffusion Atlas unifies two surfaces that most diffusion tools keep apart. The Atlas view treats diffusion as a vector process: it reads the geometry of latent space, the shape of trajectories, and the structure of neighbourhoods around a chosen point. The Bench view treats diffusion as a system to be scored: compositional fidelity (object count, attribute binding, spatial relations) measured against a controlled task pack. Atlas operations show where in the latent space a model decides; Bench operations show what those decisions cost. The two views run on the same generated images, so a finding in one can be cross-checked in the other.",
      },
      {
        title: "What is a tensor?",
        content: "Almost everything Diffusion Atlas observes is a tensor. A tensor is a multi-dimensional array of numbers with a fixed shape and a fixed numeric type (dtype). The shape tells you how the numbers are arranged. A 0-D tensor is a single number (a scalar). A 1-D tensor is a flat list (for example a CLIP text embedding of shape [768]). A 2-D tensor is a grid (a matrix, rows × columns). The latents this app traces are 4-D tensors of shape [batch, channels, height, width], typically [1, 4, 64, 64] for SD 1.5 at 512×512. That works out as one image, four latent channels, a 64×64 spatial grid, 16,384 individual floats per latent. Each step of a diffusion run produces one such tensor, and the trajectory is a sequence of them.\n\nA note on terminology. PyTorch and most ML code use \"vector\" to mean specifically a 1-D tensor. In this work I use \"vector\" in the broader mathematical sense, an element of a vector space, where every fixed-shape tensor is a vector regardless of how many axes it has. A 4-D diffusion latent of shape [1, 4, 64, 64] is, on this usage, a vector, because the set of all such tensors forms a vector space. The shape is just how you address the entries. The kind of object is the same.\n\nTensors carry a dtype (fp32, fp16, bf16, int8) which sets the precision of each number, and a device (cpu, cuda, mps) which sets where the numbers live. Changing dtype is not free: fp16 saves memory but introduces NaN bugs in older SD models on Apple Silicon, which is why the backend picks bfloat16 or fp32 per model. Changing device requires copying the tensor, and each .item() call in Python forces a host-device sync.\n\nTensors are not metaphors. They are the things that make up everything in this toolkit. The vector space is the set of all possible 4×64×64 fp32 tensors, and any one such tensor is a vector in that space. The manifold is the learned subset of vectors that the model treats as meaningful.",
      },
      {
        title: "Vector space, manifold, latent: three terms, kept separate",
        content: "Diffusion Atlas distinguishes three things that the industry term 'latent space' tends to elide. (1) Vector space — the materially-grained substrate. For SD 1.5 at 512×512 it is a 4×64×64 tensor of fp32 floats. The dimensions, the dtype, the VAE scaling factor (the magic number 0.18215) are training-and-hardware decisions, not properties of meaning. (2) Manifold — the learned geometric surface within the vector space, the Formbestimmung that the denoising network knows how to navigate. The manifold is what makes 'a courtroom without a judge' get rendered with a judge: not a fact about the substrate, a fact about the surface trained into it. (3) Latent — a single tensor in motion within the vector space, the per-step intermediate the U-Net is denoising. Plural noun, not a place. The separation is doing critical work: 'latent space' as a single term hides which of the three you're talking about, and therefore hides whether what you're observing is the substrate (a hardware contingency), the manifold (an ideological one), or a single latent in motion (a moment in a process).",
      },
      {
        title: "Why the distinction lands differently in diffusion vs LLMs",
        content: "The vector-space / manifold / latent distinction is portable across both regimes, but it is empirically more direct in diffusion. In autoregressive LLMs the manifold is *inferred*: you can probe it through embedding APIs (Manifold Atlas's tactic), but the model's actual movement happens inside transformer hidden states you don't get to see, and the output is discrete tokens, so any 'trajectory' framing is a theoretical reconstruction from the population of points. In diffusion the manifold is *traced*: the U-Net's denoising trajectory is a literal sequence of tensors, observable step by step (this is exactly what the Trajectory operation streams). The asymmetry is itself a finding. Where Manifold Atlas has to argue 'treat these points as a manifold and trust the geometry,' Diffusion Atlas can say 'watch the model walk it.' Same theoretical framework, two different empirical entry points. The term 'latent space' works as critique-resistant cover in both regimes, but for different reasons: in LLMs it elides the API-mediated *access*; in diffusion it elides the materially-graded *substrate*. The toolkit's job is to refuse the elision in both directions.",
      },
      {
        title: "Vector Logic, Continued",
        link: { label: "Vector Theory", url: "https://stunlaw.blogspot.com/2026/02/vector-theory.html" },
        content: "Diffusion Atlas extends the vector theory programme from text-embedding manifolds to generative diffusion. Where Manifold Atlas reads the static manifold of a frozen embedding model, Diffusion Atlas reads the generative trajectory of a denoising model. The asymmetry between them is itself a finding: the manifold framing migrates more cleanly to diffusion than to autoregressive text, because in diffusion the latent space is genuinely geometric rather than inferred from token logits. Position, orientation, and proximity are not metaphors here; they are the substrate on which generation runs.",
      },
    ],
  },
  {
    id: "trajectory",
    label: "Denoise Trajectory",
    entries: [
      {
        title: "What is the Denoising Trajectory?",
        content: "When a diffusion model generates an image it produces an ordered sequence of intermediate latents, one per denoising step. The sequence is the denoising trajectory: a path the model walks across its learned manifold, within the vector space its weights have organised. Reduced to three dimensions via PCA (or UMAP), the trajectory becomes a curve you can rotate and inspect. Curves that share start and end points but take different routes reveal something about how the model 'decides'. The diffusion trajectory differs from anything you can observe in a token-level LLM: in diffusion the path is *literal* — every step is a tensor that exists, can be encoded, decoded, and compared — whereas in LLMs the corresponding path through hidden states is hidden behind discrete sampling and has to be reconstructed theoretically.",
      },
      {
        title: "How the Trajectory operation works",
        content: "Run a generation against the Local FastAPI backend. The diffusers callback fires once per denoising step; the backend captures each intermediate latent (typically 4×64×64 floats for SD 1.5 at 512×512), encodes it as base64 float32 bytes, and streams it on an NDJSON line. The client decodes each line as it arrives, accumulates the latents, and shows a step counter while the stream is open. When the final 'done' event lands (along with the final image), the client projects the high-dimensional latents to three dimensions and renders the path in Three.js, or as a Film contact sheet with one frame per captured step. Start and end of the trajectory are marked in gold and burgundy respectively; the camera auto-rotates and the user can orbit, pan, and zoom. The whole run is saved to the Library so the same trajectory can be revisited or compared with later runs.",
        link: { label: "diffusers callback API", url: "https://huggingface.co/docs/diffusers/using-diffusers/callback" },
      },
      {
        title: "Per-step preview thumbnails",
        content: "The trajectory operation can decode intermediate latents through the VAE at chosen intervals. The Preview every field controls the cadence; the default of 1 decodes every step. Set it to 4 or 8 for a long run, or to 0 to skip previews entirely (saves one VAE decode per emitted thumbnail). Each thumbnail floats above its step's marker as a billboard sprite, and the Thumbnails every slider thins the swarm so the curve stays readable. The final step is always thumbnailed regardless of the cadence so the destination is visible. Decoding cost on Apple Silicon is roughly 0.5–1 second per thumbnail at 96 px; on CUDA, faster.",
      },
      {
        title: "PCA vs UMAP for the trajectory projection",
        content: "Why do PCA and UMAP draw different curves from the same trajectory? Because they answer different questions about the data. PCA is a linear projection onto the three directions of greatest variance: it preserves global distances (long stretches of the trajectory keep their relative scale) and is deterministic. UMAP is a non-linear, neighbourhood-graph-based method: it tries to preserve which points are *close to which*, even at the cost of distorting how *far apart* clusters are. PCA reads the trajectory as a path through Euclidean space; UMAP reads it as a topology. For a fairly direct denoising run the two views often look broadly similar; for a trajectory that doubles back, switches basins, or curves sharply, they diverge — and that divergence is itself information about local manifold curvature. Practical notes: PCA is fast (run live as steps stream); UMAP is slower and stochastic, so different runs of UMAP on the same data may produce different rotations or warpings. UMAP needs ≥4 samples; below that the toggle silently falls back to PCA. The disagreement between the two views — read carefully — is one of the few places where the *shape* of the manifold leaks out, rather than just its coordinates.",
        link: { label: "umap-js", url: "https://github.com/PAIR-code/umap-js" },
      },
      {
        title: "Temporary and locked layers",
        content: "Every completed run is added to the layers list as a temporary layer (dashed border, italic label). Running again replaces it. Click the padlock to lock a layer: locked layers survive later runs and take a palette colour so several trajectories can be overlaid. The seed input has shuffle and increment toggles next to the dice: shuffle rolls a new random seed before each run; increment adds 1 before each run.",
      },
    ],
  },
  {
    id: "sweep",
    label: "Guidance Sweep",
    entries: [
      {
        title: "What is Classifier-Free Guidance?",
        link: { label: "Ho and Salimans (2022)", url: "https://arxiv.org/abs/2207.12598" },
        content: "Classifier-Free Guidance (CFG) is the lever that controls how strongly the model follows the text prompt. At CFG = 1 the model ignores the prompt entirely. At higher values it amplifies the difference between the prompt-conditioned and unconditioned predictions, pushing the trajectory more aggressively toward the prompt. Too low and the output drifts from the prompt; too high and the output collapses into oversaturated, mode-collapsed images. The Guidance Sweep operation traces this controllability surface and reveals where each model becomes brittle.",
      },
      {
        title: "Running a sweep",
        content: "Guidance Sweep generates the same prompt and seed at each CFG value in the list (default 1, 4, 7.5, 12, 18). Cells fill in as they finish. The sweep runs one cell at a time per lane, so one failure does not stop the run, and rate-limited responses show a Retrying in Ns countdown before resuming.",
      },
      {
        title: "Drift curve",
        content: "When the sweep finishes, each image is hashed (16×16 grayscale aHash, 256 bits) in the browser. Hamming distance between two hashes divided by 256 gives a drift score in [0, 1]. The baseline is the CFG value nearest 7.5, so drift reads as 'how far does this CFG value pull the image away from the conventional default?'. Low CFG drifts because the image has wandered off the prompt; high CFG drifts because the image has collapsed into oversaturated territory. The curve is in the Deep Dive panel and in the PDF export. aHash is a coarse proxy for visual difference; the CLIP scorer on the local backend is used by Compositional Bench, not by the drift curve.",
      },
      {
        title: "Cross-backend comparison",
        content: "Tick Compare with a second provider and pick one. The same prompt, seed and CFG list run against both providers in parallel. Two grids stack with their drift curves; geometry present in both separates from geometry contingent on one model's training. The lanes have independent rate limits and retry loops, so a 429 in one does not block the other. The most informative pairing is hosted against local: the same prompt at the same seed on Replicate's flux-schnell and on a local SD 1.5 shows the cost of a smaller, older model on the same task. Latent Neighbourhood and Compositional Bench have the same Compare option.",
      },
    ],
  },
  {
    id: "neighbourhood",
    label: "Latent Neighbourhood",
    entries: [
      {
        title: "Sampling around an anchor",
        content: "Latent Neighbourhood generates k images around an anchor seed, using seed offsets spread evenly across a chosen radius. The offsets are deterministic, so a run can be repeated exactly. Each seed maps to a different starting tensor in the model's vector space, so each image comes from a different trajectory across the manifold.",
      },
      {
        title: "Reading the grid",
        content: "Nearby seeds that look similar indicate a smooth basin in the manifold; a sudden categorical jump (a cat becomes a dog) marks a basin boundary. A small radius probes local geometry; a large radius probes how connected the manifold is at distance. Gaussian perturbation of the initial latent at a chosen σ, a tighter probe, needs a /neighbourhood endpoint on the local backend that has not been built yet.",
      },
      {
        title: "Cross-backend comparison",
        content: "Tick Compare with a second provider to run the same anchor and seed offsets against another provider in parallel. Where the two neighbourhoods cluster differently, the geometry depends on the model rather than on diffusion as such.",
      },
    ],
  },
  {
    id: "bench",
    label: "Compositional Bench",
    entries: [
      {
        title: "What is GenEval?",
        link: { label: "Ghosh et al. (2023)", url: "https://arxiv.org/abs/2310.11513" },
        content: "GenEval is a benchmark for compositional fidelity in text-to-image models. Rather than scoring overall aesthetic quality, it evaluates whether the model has correctly bound attributes ('a red cube and a blue sphere'), preserved counts ('three apples'), or honoured spatial relations ('a cat to the left of a dog'). These are the classical compositionality tests that humans pass effortlessly and that diffusion models still fail in characteristic ways. The Compositional Bench in Diffusion Atlas runs a GenEval-lite task pack (four categories, three prompts each) against any configured provider and reports per-category accuracy with per-prompt detail in the deep-dive panel.",
      },
      {
        title: "CLIP auto-scoring",
        content: "The Auto-score (CLIP) button next to Run bench posts every generated image plus its prompt to the local backend's /score endpoint, which loads openai/clip-vit-base-patch32 (~150 MB) on first use and returns per-pair cosine similarity. Each task card shows the score; verdicts are set against a threshold (default 0.25, configurable in the toolbar). The threshold is exposed because what counts as 'matching' is a methodological choice: 0.20 is permissive, 0.30 is strict. Manual pass/fail still works on top. CLIP cosine is a coarse proxy for compositional fidelity; it does not check 'exactly two objects' or colour binding directly. Object detection and attribute-binding analysis along GenEval's original recipe have not been built yet. Auto-scoring needs the local backend running even when the images came from a hosted provider.",
        link: { label: "GenEval", url: "https://arxiv.org/abs/2310.11513" },
      },
      {
        title: "Cross-backend comparison",
        content: "Tick Compare with a second provider to run the same task pack against another provider and compare per-category scores side by side.",
      },
    ],
  },
  {
    id: "backends",
    label: "Backends and models",
    entries: [
      {
        title: "Hosted vs Local Backend",
        content: "Hosted providers (Replicate and Fal) are the easiest path to running a generation: you supply an API key and a prompt, they return an image. They do not expose intermediate latents, and that closes off most of the Atlas view. The local backend runs the diffusers library on your own hardware via a small FastAPI service. It is heavier to set up, requires a GPU (or Apple Silicon), and is the only way to run Denoise Trajectory. Diffusion Atlas treats the two as peer providers selected in Settings; capability mismatches surface as typed errors that suggest switching. Together and Stability AI are planned but not wired up.",
      },
      {
        title: "Choosing a hosted provider",
        content: "Replicate has the broader model catalogue (SD, SDXL, SD3, FLUX, plus thousands of community models) and is the conventional starting point. Fal is the better choice for sweep- and bench-heavy work because its rate limits are typically an order of magnitude more permissive: a 12-task bench that runs into 429s on Replicate's free tier runs cleanly on Fal. Switching providers in Settings resets the model id to that provider's default, since the naming conventions differ (Replicate: owner/model; Fal: fal-ai/<model>).",
      },
      {
        title: "Local models and Hugging Face",
        content: "The local backend downloads weights from Hugging Face on first use and caches them in ~/.cache/huggingface/. Stable Diffusion 1.5 (stable-diffusion-v1-5/stable-diffusion-v1-5), SDXL and SDXL-turbo are ungated. SD 3 / 3.5 and both FLUX.1 models are gated: accept the licence on the model page, then run huggingface-cli login in the backend's virtual environment.",
        link: { label: "Hugging Face setup", url: `${DOCS_URL}#api-keys` },
      },
      {
        title: "What is MPS warmup?",
        content: "When you switch the backend to Local on an Apple Silicon machine, the StatusBar dot says \"warming\" for a minute or two and then flips to \"warm\". That delay is real. It exists because PyTorch's MPS (Metal Performance Shaders) backend pays five separate first-time costs the first time it runs a diffusion model:\n\n1. Kernel compilation. Every unique combination of (operation, input shape, dtype) triggers Metal Shading Language to JIT-compile a GPU shader. SD 1.5 has dozens of unique convolutions, attention layers, group norms, etc. Each takes 50–500 ms. The first inference compiles them all serially — ~10–30 seconds spent in MTLDevice.makeComputePipelineState.\n\n2. MPS allocator initialisation. PyTorch's MPS allocator carves heap regions on first use and tunes its block sizes empirically. The first dozen allocations are slower than the next thousand because the allocator hasn't yet figured out a sensible fragmentation strategy.\n\n3. Weight paging. Model weights live in mmap'd safetensors files. The first forward physically reads them into unified memory page-by-page. SD 1.5 fp32 ≈ 5 GB; that's a lot of disk pages to fault in. Subsequent forwards already have them resident.\n\n4. Metal library load + command queue priming. macOS lazily mmaps the Metal Performance Shaders kernel library on first use. First GPU command submission also has higher latency than subsequent ones — the kernel scheduling path itself has cold-cache effects.\n\n5. Driver-level command-buffer caching. Apple's GPU driver caches recently-issued command buffers and reuses their dispatch metadata. The first run sees zero cache hits; the second sees most.\n\nConcrete numbers on an M5 MacBook Pro: cold first step ≈ 80 seconds. Warmed second step ≈ 1–3 seconds. So a 12-step trajectory cold takes ~16 minutes; warm takes ~30 seconds. Same model, same seed — the only difference is whether those five caches are populated.\n\nWhat the /warmup endpoint does: runs one denoising step at 256×256 with prompt=\"warmup\" and CFG 0. Throws away the image. The point is the side effects — all five caches above are now warm. A 256² warmup amortises maybe 70–80% of the cold cost. The first real 512² run still pays a small (~5–10 s) tax for shape-specialised kernel recompiles, but that's an order of magnitude better than the full cold start.\n\nWhy does this exist on MPS specifically? CUDA has the same cold-start problem but it's faster because (a) NVIDIA's compiler (PTX → SASS) is more mature and pre-caches more aggressively, (b) CUDA has had compiled-kernel caching infrastructure for fifteen years, (c) NVIDIA driver-level tooling pre-loads what's likely to be needed. Apple's MPS backend in PyTorch shipped in 2022 and is still catching up; the JIT compilation is more visible because there's less amortisation across the framework. Treat the warmup as a tax for running diffusion locally on a machine the framework is still learning to use well.",
      },
    ],
  },
  {
    id: "tips",
    label: "Tips",
    entries: [
      {
        title: "Fast mode",
        content: "Fast mode in Settings caps the step count of every operation, whatever the form says. Use it to iterate on prompts and layout before a full-quality run.",
      },
      {
        title: "Stop, inspect, export",
        content: "Every operation has a Stop button. Clicking any image opens it at full size with an RGB histogram and image statistics. Runs export to PDF, and the film contact sheets to SVG and PNG.",
      },
      {
        title: "Library",
        content: "Every saved run from Sweep, Neighbourhood, Bench and Trajectory is listed in the Library tab, grouped by kind, newest first. Runs are stored in this browser's IndexedDB and never leave the machine. Each run can be deleted, or the whole Library cleared.",
      },
      {
        title: "Why cache latents and images?",
        content: "Diffusion is expensive: a single trajectory at 30 steps and 1024×1024 can take 5 to 30 seconds depending on hardware. Diffusion Atlas caches latents (Float32Array) and images (Blob) in IndexedDB, keyed by the full generation parameters (model, prompt, seed, steps, CFG), so identical queries return at once. The image cache uses LRU eviction with a cap (default 500 MB). Settings shows the cache counts and can clear either cache.",
      },
    ],
  },
];

export function HelpModal() {
  const [open, setOpen] = useState(false);
  const [pane, setPane] = useState(HELP_PANES[0].id);
  useEscape(open, () => setOpen(false));

  const active = HELP_PANES.find((p) => p.id === pane) ?? HELP_PANES[0];

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="btn-editorial-ghost px-3 py-2"
        title="Help: Diffusion model concepts"
      >
        <HelpCircle size={16} />
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-4"
          onClick={() => setOpen(false)}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="help-title"
            className="card-editorial shadow-editorial-lg w-full max-w-4xl h-[85vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-6 py-4 border-b border-parchment flex items-center justify-between shrink-0">
              <div className="flex items-center gap-3">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/icons/vector-lab-diffusion-atlas.svg"
                  alt=""
                  width={28}
                  height={28}
                  aria-hidden="true"
                  className="block shrink-0"
                />
                <h2 id="help-title" className="font-display text-display-md font-bold">
                  Diffusion Atlas Guide
                </h2>
              </div>
              <button onClick={() => setOpen(false)} className="btn-editorial-ghost px-2 py-1" aria-label="Close help">
                <X size={16} />
              </button>
            </div>

            <div className="flex flex-col md:flex-row flex-1 min-h-0">
              <nav
                className="flex md:flex-col gap-1 p-3 md:w-52 shrink-0 overflow-x-auto md:overflow-y-auto border-b md:border-b-0 md:border-r border-parchment"
                aria-label="Help sections"
              >
                {HELP_PANES.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => setPane(p.id)}
                    className={`text-left whitespace-nowrap px-3 py-1.5 rounded-sm font-sans text-body-sm transition-colors ${
                      pane === p.id
                        ? "bg-burgundy/10 text-burgundy font-semibold"
                        : "text-muted-foreground hover:text-foreground hover:bg-cream/50"
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
              </nav>

              <div className="flex-1 min-h-0 overflow-y-auto px-6 py-5 space-y-6">
                {active.entries.map((entry) => (
                  <section key={entry.title}>
                    <h3 className="font-display text-body-lg font-semibold mb-2">{entry.title}</h3>
                    <div className="font-body text-body-sm text-slate leading-relaxed whitespace-pre-line">
                      {entry.content}
                    </div>
                    {entry.link && (
                      <p className="mt-2">
                        <a
                          href={entry.link.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-sans text-caption text-burgundy underline underline-offset-2 hover:text-burgundy-900"
                        >
                          {entry.link.label} &rarr;
                        </a>
                      </p>
                    )}
                  </section>
                ))}
              </div>
            </div>

            <div className="px-6 py-3 border-t border-parchment flex flex-wrap items-center gap-x-3 gap-y-1 shrink-0 font-sans text-caption text-muted-foreground">
              <a href={`${DOCS_URL}#faq`} target="_blank" rel="noopener noreferrer" className="hover:text-foreground">
                Full guide and FAQ
              </a>
              <span aria-hidden="true">·</span>
              <a
                href="https://vector-lab-tools.github.io"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 hover:text-foreground"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src="/icons/vector-lab-logo-mark.svg" alt="" width={14} height={14} className="w-3.5 h-3.5" />
                Part of the Vector Lab
              </a>
              <span aria-hidden="true">·</span>
              <a
                href="https://github.com/vector-lab-tools/diffusionatlas"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-foreground"
              >
                GitHub
              </a>
              <span aria-hidden="true">·</span>
              <a
                href="https://stunlaw.blogspot.com/2026/02/vector-theory.html"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-foreground"
              >
                Vector Theory
              </a>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
