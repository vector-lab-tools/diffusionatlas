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
    title: "What is Latent Space?",
    content: "Modern diffusion models do not denoise raw pixels. They operate on a compressed latent representation produced by an encoder (typically a VAE), then decode the final latent back to an image. Latent space is therefore where the generative work actually happens. Each point in this space corresponds to a possible image, and small movements in latent space tend to produce smooth changes in the output. Diffusion Atlas studies the geometry of this space: where models concentrate, where they thin out, and where small perturbations produce categorical jumps.",
  },
  {
    title: "What is the Denoising Trajectory?",
    content: "When a diffusion model generates an image it produces an ordered sequence of intermediate latents, one per denoising step. This sequence is the denoising trajectory. Reduced to three dimensions via UMAP or PCA, the trajectory becomes a curve through latent space. Curves that share start and end points but take different routes reveal something about how the model 'decides'. The trajectory is the analogue, for diffusion, of the token trajectory in autoregressive language models: a path through a learned manifold that ends in a specific output.",
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
                      <p className="font-body text-body-sm text-slate leading-relaxed">
                        {section.content}
                      </p>
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
