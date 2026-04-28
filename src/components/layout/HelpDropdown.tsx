"use client";

import { useRef, useState, useEffect } from "react";
import { HelpCircle, X, ChevronRight, ChevronDown } from "lucide-react";

interface HelpSection {
  title: string;
  content: string;
  link?: { label: string; url: string };
}

const HELP_SECTIONS: HelpSection[] = [
  {
    title: "What is a Diffusion Model?",
    content: "A diffusion model generates an image by reversing a noise process. It starts with pure noise and, over a fixed number of steps, denoises that noise into a coherent image. At each step the model predicts what to remove, conditioned on a text prompt. The trajectory through this denoising process is the object of study for Diffusion Atlas: the same final image can be reached by very different paths, and what the model 'decides' along the way is often more revealing than the picture it ends on.",
  },
  {
    title: "What is MPS warmup?",
    content: "When you switch the backend to Local on an Apple Silicon machine, the StatusBar dot says \"warming\" for a minute or two and then flips to \"warm\". That delay is real. It exists because PyTorch's MPS (Metal Performance Shaders) backend pays five separate first-time costs the first time it runs a diffusion model:\n\n1. Kernel compilation. Every unique combination of (operation, input shape, dtype) triggers Metal Shading Language to JIT-compile a GPU shader. SD 1.5 has dozens of unique convolutions, attention layers, group norms, etc. Each takes 50–500 ms. The first inference compiles them all serially — ~10–30 seconds spent in MTLDevice.makeComputePipelineState.\n\n2. MPS allocator initialisation. PyTorch's MPS allocator carves heap regions on first use and tunes its block sizes empirically. The first dozen allocations are slower than the next thousand because the allocator hasn't yet figured out a sensible fragmentation strategy.\n\n3. Weight paging. Model weights live in mmap'd safetensors files. The first forward physically reads them into unified memory page-by-page. SD 1.5 fp32 ≈ 5 GB; that's a lot of disk pages to fault in. Subsequent forwards already have them resident.\n\n4. Metal library load + command queue priming. macOS lazily mmaps the Metal Performance Shaders kernel library on first use. First GPU command submission also has higher latency than subsequent ones — the kernel scheduling path itself has cold-cache effects.\n\n5. Driver-level command-buffer caching. Apple's GPU driver caches recently-issued command buffers and reuses their dispatch metadata. The first run sees zero cache hits; the second sees most.\n\nConcrete numbers on an M5 MacBook Pro: cold first step ≈ 80 seconds. Warmed second step ≈ 1–3 seconds. So a 12-step trajectory cold takes ~16 minutes; warm takes ~30 seconds. Same model, same seed — the only difference is whether those five caches are populated.\n\nWhat the /warmup endpoint does: runs one denoising step at 256×256 with prompt=\"warmup\" and CFG 0. Throws away the image. The point is the side effects — all five caches above are now warm. Honest accounting: a 256² warmup amortises maybe 70–80% of the cold cost. The first real 512² run still pays a small (~5–10 s) tax for shape-specialised kernel recompiles, but that's an order of magnitude better than the full cold start.\n\nWhy does this exist on MPS specifically? CUDA has the same cold-start problem but it's faster because (a) NVIDIA's compiler (PTX → SASS) is more mature and pre-caches more aggressively, (b) CUDA has had compiled-kernel caching infrastructure for fifteen years, (c) NVIDIA driver-level tooling pre-loads what's likely to be needed. Apple's MPS backend in PyTorch shipped in 2022 and is still catching up; the JIT compilation is more visible because there's less amortisation across the framework. Treat the warmup as a tax for running diffusion locally on a machine the framework is still learning to use well.",
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
    title: "What is the Denoising Trajectory?",
    content: "When a diffusion model generates an image it produces an ordered sequence of intermediate latents, one per denoising step. The sequence is the denoising trajectory: a path the model walks across its learned manifold, within the vector space its weights have organised. Reduced to three dimensions via PCA (or UMAP), the trajectory becomes a curve you can rotate and inspect. Curves that share start and end points but take different routes reveal something about how the model 'decides'. The diffusion trajectory differs from anything you can observe in a token-level LLM: in diffusion the path is *literal* — every step is a tensor that exists, can be encoded, decoded, and compared — whereas in LLMs the corresponding path through hidden states is hidden behind discrete sampling and has to be reconstructed theoretically.",
  },
  {
    title: "How the Trajectory operation works",
    content: "Run a generation against the Local FastAPI backend. The diffusers callback fires once per denoising step; the backend captures each intermediate latent (typically 4×64×64 floats for SD 1.5 at 512×512), encodes it as base64 float32 bytes, and streams it on an NDJSON line. The client decodes each line as it arrives, accumulates the latents, and shows a step counter while the stream is open. When the final 'done' event lands (along with the final image), the client runs PCA in the browser to project the high-dimensional latents to three dimensions and renders the path in Three.js. Start and end of the trajectory are marked in gold and burgundy respectively; the camera auto-rotates and the user can orbit, pan, and zoom. The whole run is saved to the Library so the same trajectory can be revisited or compared with later runs.",
    link: { label: "diffusers callback API", url: "https://huggingface.co/docs/diffusers/using-diffusers/callback" },
  },
  {
    title: "Per-step preview thumbnails",
    content: "Since v0.2.2 the trajectory operation can decode intermediate latents through the VAE at chosen intervals. The Preview every field controls the cadence: every 4th step is a reasonable default for a 20-step run; set it to 0 to skip previews entirely (saves ~one VAE decode per emitted thumbnail). Each thumbnail floats above its step's marker as a billboard sprite, so you can see the image emerging from noise along the actual trajectory through latent space. The final step is always thumbnailed regardless of the cadence so the destination is visible. Decoding cost on Apple Silicon is roughly 0.5–1 second per thumbnail at 96 px; on CUDA, faster. The cost is in addition to denoising itself, so a trajectory with previewEvery=2 takes noticeably longer than one with previewEvery=8.",
  },
  {
    title: "Cross-backend comparison",
    content: "Diffusion Atlas v0.3.0 adds a Compare-with toggle to Guidance Sweep. Tick it, pick a second provider, and the same prompt + seed + CFG list runs against both providers in parallel. Two grids stack with their drift curves; side-by-side, geometry that is structural — present in both — separates from geometry that is contingent on one model's training. The two lanes have independent rate limits and independent retry loops, so a 429 in one doesn't block the other. The most informative pairing is hosted ↔ local: an identical prompt at the same seed across (say) Replicate's flux-schnell and your local SD 1.5 makes very visible the cost of a smaller, older model on the same compositional task. This is the move that distinguishes Diffusion Atlas from a single-provider tool: comparison is the methodological precondition for separating regime from instance. The same comparison shape is queued for Latent Neighbourhood and Compositional Bench.",
  },
  {
    title: "Choosing a hosted provider",
    content: "Diffusion Atlas wires both Replicate and Fal.ai. Replicate has the broader model catalogue (SD, SDXL, SD3, FLUX, plus thousands of community models) and is the conventional starting point. Fal.ai is the better choice for sweep- and bench-heavy work because its rate limits are typically an order of magnitude more permissive — a 12-task bench that thrashes with 429s on Replicate's free tier runs cleanly on Fal. Both use the same DiffusionProvider interface so operations don't care which one is in play; only the input shape and authentication header differ. Switching providers in Settings auto-updates the model id to a sensible default for that provider since the naming conventions diverge: Replicate uses `owner/model`, Fal uses `fal-ai/<model>`. Override the model field afterwards if you want something specific.",
  },
  {
    title: "PCA vs UMAP for the trajectory projection",
    content: "Why do PCA and UMAP draw different curves from the same trajectory? Because they answer different questions about the data. PCA is a linear projection onto the three directions of greatest variance: it preserves global distances (long stretches of the trajectory keep their relative scale) and is deterministic. UMAP is a non-linear, neighbourhood-graph-based method: it tries to preserve which points are *close to which*, even at the cost of distorting how *far apart* clusters are. PCA reads the trajectory as a path through Euclidean space; UMAP reads it as a topology. For a fairly direct denoising run the two views often look broadly similar; for a trajectory that doubles back, switches basins, or curves sharply, they diverge — and that divergence is itself information about local manifold curvature. Practical notes: PCA is fast (run live as steps stream); UMAP is slower and stochastic, so different runs of UMAP on the same data may produce different rotations or warpings. UMAP needs ≥4 samples; below that the toggle silently falls back to PCA. The disagreement between the two views — read carefully — is one of the few places where the *shape* of the manifold leaks out, rather than just its coordinates.",
    link: { label: "umap-js", url: "https://github.com/PAIR-code/umap-js" },
  },
  {
    title: "CLIP auto-scoring in Compositional Bench",
    content: "Since v0.2.3 the bench has an Auto-score (CLIP) button next to Run bench. It posts every generated image plus its source prompt to the local backend's /score endpoint, which lazily loads openai/clip-vit-base-patch32 (~150 MB) and returns per-pair cosine similarity in the conventional CLIP scale. Each task card shows the numeric score next to the prompt; verdicts are set automatically against a threshold (default 0.25, configurable in the toolbar). The threshold is intentionally exposed because what counts as 'matching' is a methodological choice, not a fact: 0.20 is permissive, 0.30 is strict. Manual pass/fail still works on top — auto-score sets defaults you can override per card. CLIP cosine is a coarse proxy for compositional fidelity; it does not directly check 'exactly two objects' or 'red cube vs blue sphere binding'. More principled scoring (object detection, attribute-binding analysis along GenEval's original recipe) is queued.",
    link: { label: "GenEval", url: "https://arxiv.org/abs/2310.11513" },
  },
  {
    title: "Drift curve in Guidance Sweep",
    content: "When the sweep finishes, each image is hashed (16×16 grayscale aHash, 256 bits) entirely in the browser via Canvas. Hamming distance between any two hashes is divided by 256 to give a normalised drift score in [0, 1]. The baseline is the CFG value nearest 7.5 — the conventional 'sensible default' for SDXL/SD3 — so drift reads as 'how far does this CFG value pull the image away from the conventionally-correct version?'. Low CFG drifts because the image has wandered off the prompt; high CFG drifts because the image has collapsed into oversaturated mode-collapsed territory. The curve is the controllability surface, made literal. aHash is a coarse proxy for visual difference — for a more rigorous metric, a CLIP-based score endpoint is queued for the local backend.",
  },
  {
    title: "What is Classifier-Free Guidance?",
    link: { label: "Ho and Salimans (2022)", url: "https://arxiv.org/abs/2207.12598" },
    content: "Classifier-Free Guidance (CFG) is the lever that controls how strongly the model follows the text prompt. At CFG = 1 the model ignores the prompt entirely. At higher values it amplifies the difference between the prompt-conditioned and unconditioned predictions, pushing the trajectory more aggressively toward the prompt. Too low and the output drifts from the prompt; too high and the output collapses into oversaturated, mode-collapsed images. The Guidance Sweep operation traces this controllability surface and reveals where each model becomes brittle.",
  },
  {
    title: "Atlas vs Bench",
    content: "Diffusion Atlas unifies two surfaces that most diffusion tools keep apart. The Atlas view treats diffusion as a vector process: it reads the geometry of latent space, the shape of trajectories, and the structure of neighbourhoods around a chosen point. The Bench view treats diffusion as a system to be scored: compositional fidelity (object count, attribute binding, spatial relations) measured against a controlled task pack. Atlas operations show where in the latent space a model decides; Bench operations show what those decisions cost. The two views run on the same generated images, so a finding in one can be cross-checked in the other.",
  },
  {
    title: "Hosted vs Local Backend",
    content: "Hosted providers (Replicate, Fal, Together, Stability) are the easiest path to running a generation: you supply an API key and a prompt, they return an image. They do not, however, expose intermediate latents, and that closes off most of the Atlas view. The local backend runs the diffusers library on your own hardware via a small FastAPI service. It is heavier to set up, requires a GPU (or Apple Silicon), and is the only way to run Denoise Trajectory or true latent-space neighbourhood sampling. Diffusion Atlas treats the two as peer providers selected via Settings; capability mismatches surface as typed errors that suggest switching.",
  },
  {
    title: "What is GenEval?",
    link: { label: "Ghosh et al. (2023)", url: "https://arxiv.org/abs/2310.11513" },
    content: "GenEval is a benchmark for compositional fidelity in text-to-image models. Rather than scoring overall aesthetic quality, it evaluates whether the model has correctly bound attributes ('a red cube and a blue sphere'), preserved counts ('three apples'), or honoured spatial relations ('a cat to the left of a dog'). These are the classical compositionality tests that humans pass effortlessly and that diffusion models still fail in characteristic ways. The Compositional Bench in Diffusion Atlas runs a GenEval-lite task pack against any configured provider and reports per-category accuracy with per-prompt detail in the deep-dive panel.",
  },
  {
    title: "Why Cache Latents and Images?",
    content: "Diffusion is expensive: a single trajectory at 30 steps and 1024x1024 resolution can take 5 to 30 seconds depending on hardware. Re-running the same prompt at the same seed should never cost twice. Diffusion Atlas caches latents (Float32Array) and images (Blob) in IndexedDB, keyed by the full generation parameters (model, prompt, seed, steps, cfg). Identical queries return instantly. Image blobs dwarf text embeddings, so the image cache uses LRU eviction with a configurable cap (default 500 MB).",
  },
  {
    title: "Vector Logic, Continued",
    link: { label: "Vector Theory", url: "https://stunlaw.blogspot.com/2026/02/vector-theory.html" },
    content: "Diffusion Atlas extends the vector theory programme from text-embedding manifolds to generative diffusion. Where Manifold Atlas reads the static manifold of a frozen embedding model, Diffusion Atlas reads the generative trajectory of a denoising model. The asymmetry between them is itself a finding: the manifold framing migrates more cleanly to diffusion than to autoregressive text, because in diffusion the latent space is genuinely geometric rather than inferred from token logits. Position, orientation, and proximity are not metaphors here; they are the substrate on which generation runs.",
  },
];

export function HelpDropdown() {
  const [open, setOpen] = useState(false);
  const [expandedSection, setExpandedSection] = useState<number | null>(null);
  const sectionRefs = useRef<Array<HTMLDivElement | null>>([]);

  useEffect(() => {
    if (expandedSection === null) return;
    const el = sectionRefs.current[expandedSection];
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [expandedSection]);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="btn-editorial-ghost px-3 py-2"
        title="Help: Diffusion model concepts"
      >
        <HelpCircle size={16} />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div
            className="fixed top-4 right-4 z-50 w-[640px] max-w-[calc(100vw-2rem)] card-editorial shadow-editorial-lg flex flex-col"
            style={{ maxHeight: "calc(100vh - 2rem)" }}
          >
            <div className="p-4 border-b border-parchment flex items-center justify-between flex-shrink-0">
              <div className="flex items-start gap-3">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/icons/vector-lab-diffusion-atlas.svg"
                  alt=""
                  width={32}
                  height={32}
                  aria-hidden="true"
                  className="block flex-shrink-0 mt-0.5"
                />
                <div>
                  <h2 className="font-display text-display-md font-bold">Diffusion Atlas Guide</h2>
                  <p className="font-sans text-caption text-muted-foreground mt-0.5">
                    Key concepts for reading diffusion models geometrically
                  </p>
                  <p className="font-sans text-caption text-muted-foreground mt-1">
                    Part of the{" "}
                    <a
                      href="https://vector-lab-tools.github.io"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-burgundy underline underline-offset-2 hover:text-burgundy-900"
                    >
                      Vector Lab
                    </a>
                  </p>
                </div>
              </div>
              <button onClick={() => setOpen(false)} className="btn-editorial-ghost px-2 py-1">
                <X size={16} />
              </button>
            </div>

            <div className="divide-y divide-parchment flex-1 overflow-y-auto">
              {HELP_SECTIONS.map((section, i) => (
                <div
                  key={i}
                  ref={(el) => {
                    sectionRefs.current[i] = el;
                  }}
                >
                  <button
                    onClick={() => setExpandedSection(expandedSection === i ? null : i)}
                    className="w-full text-left px-4 py-3 flex items-center gap-2 hover:bg-cream/50 transition-colors"
                  >
                    {expandedSection === i ? (
                      <ChevronDown size={14} className="text-burgundy" />
                    ) : (
                      <ChevronRight size={14} className="text-muted-foreground" />
                    )}
                    <span className="font-sans text-body-sm font-semibold">{section.title}</span>
                  </button>
                  {expandedSection === i && (
                    <div className="px-4 pb-4 pl-10">
                      <div className="font-body text-body-sm text-slate leading-relaxed space-y-2 whitespace-pre-line">
                        {section.content}
                      </div>
                      {section.link && (
                        <p className="mt-2">
                          <a
                            href={section.link.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="font-sans text-caption text-burgundy underline underline-offset-2 hover:text-burgundy-900"
                          >
                            {section.link.label} &rarr;
                          </a>
                        </p>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="p-4 border-t border-parchment flex-shrink-0">
              <p className="font-sans text-caption text-muted-foreground">
                Based on{" "}
                <a
                  href="https://stunlaw.blogspot.com/2026/02/vector-theory.html"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-burgundy underline"
                >
                  Vector Theory
                </a>{" "}
                by David M. Berry.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
