"use client";

import { useState } from "react";
import { Info, X } from "lucide-react";
import { VERSION, VERSION_DATE } from "@/lib/version";

export function AboutModal() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="btn-editorial-ghost px-3 py-2"
        title="About Diffusion Atlas"
      >
        <Info size={16} />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 bg-black/30 z-50" onClick={() => setOpen(false)} />
          <div
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-[560px] max-w-[calc(100vw-2rem)] card-editorial shadow-editorial-lg animate-fade-in flex flex-col"
            style={{ maxHeight: "calc(100vh - 2rem)" }}
          >
            <div className="px-6 pt-6 pb-4 flex items-start justify-between flex-shrink-0">
              <div className="flex items-start gap-3">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/icons/vector-lab-diffusion-atlas.svg"
                  alt=""
                  width={40}
                  height={40}
                  aria-hidden="true"
                  className="block flex-shrink-0 mt-1"
                />
                <div>
                  <h2 className="font-display text-display-lg font-bold text-burgundy">Diffusion Atlas</h2>
                  <p className="font-sans text-caption text-muted-foreground mt-0.5">
                    Manifold geometry and benchmarking for diffusion models
                  </p>
                </div>
              </div>
              <button onClick={() => setOpen(false)} className="btn-editorial-ghost px-2 py-1">
                <X size={16} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto">
              <div className="px-6 py-4 space-y-3 border-t border-parchment">
                <div className="grid grid-cols-[120px_1fr] gap-y-2 font-sans text-body-sm">
                  <span className="text-muted-foreground">Version</span>
                  <span className="font-medium">{VERSION}</span>

                  <span className="text-muted-foreground">Date</span>
                  <span className="font-medium">{VERSION_DATE}</span>

                  <span className="text-muted-foreground">Author</span>
                  <span className="font-medium">David M. Berry</span>

                  <span className="text-muted-foreground">Affiliation</span>
                  <span className="font-medium">University of Sussex</span>

                  <span className="text-muted-foreground">Implemented with</span>
                  <span className="font-medium">Claude Code</span>

                  <span className="text-muted-foreground">Design system</span>
                  <span className="font-medium">CCS-WB Editorial</span>

                  <span className="text-muted-foreground">Licence</span>
                  <span className="font-medium">MIT</span>
                </div>
              </div>

              <div className="px-6 py-4 border-t border-parchment">
                <p className="font-body text-body-sm text-slate leading-relaxed">
                  Diffusion Atlas is a research instrument for the geometry of generative
                  diffusion models. Where{" "}
                  <a
                    href="https://github.com/vector-lab-tools/manifold-atlas"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-burgundy underline underline-offset-2"
                  >
                    Manifold Atlas
                  </a>{" "}
                  reads the static manifold of a frozen embedding model, Diffusion Atlas reads the
                  generative trajectory of a denoising model: the path through latent space that
                  produces an image. It unifies two surfaces that current diffusion tools keep
                  apart: an <strong>Atlas</strong> view (interpretability and geometry, per-step
                  latents, CFG sweeps, neighbourhood sampling) and a <strong>Bench</strong> view
                  (scored compositional evaluation in the GenEval tradition).
                </p>
                <p className="font-body text-body-sm text-slate leading-relaxed mt-3">
                  The unification is theoretical, not cosmetic. Atlas operations expose where in
                  latent space a model decides; Bench shows what those decisions cost in
                  compositional fidelity. Together they let critical theorists treat diffusion
                  outputs as traces of a vector logic rather than finished pictures.
                </p>
              </div>

              <div className="px-6 py-4 border-t border-parchment">
                <h3 className="font-sans text-caption text-muted-foreground uppercase tracking-wider font-semibold mb-2">
                  Part of the Vector Lab
                </h3>
                <div className="flex items-start gap-3">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src="/icons/vector-lab-logo-mark.svg"
                    alt=""
                    width={32}
                    height={32}
                    aria-hidden="true"
                    className="block flex-shrink-0 mt-0.5 opacity-90"
                  />
                  <div className="font-body text-body-sm text-slate leading-relaxed">
                    <p>
                      Diffusion Atlas is one of the research instruments in the{" "}
                      <a
                        href="https://vector-lab-tools.github.io"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-burgundy underline underline-offset-2 hover:text-burgundy-900"
                      >
                        Vector Lab
                      </a>
                      . <strong>Diffusion Atlas reads the generative trajectory.</strong>
                    </p>
                    <p className="mt-2 font-sans text-caption">
                      Sibling instruments:{" "}
                      <a href="https://github.com/vector-lab-tools/manifold-atlas" target="_blank" rel="noopener noreferrer" className="text-burgundy underline underline-offset-2">Manifold Atlas</a>{" · "}
                      <a href="https://github.com/vector-lab-tools/LLMbench" target="_blank" rel="noopener noreferrer" className="text-burgundy underline underline-offset-2">LLMbench</a>{" · "}
                      <a href="https://github.com/vector-lab-tools/vectorscope" target="_blank" rel="noopener noreferrer" className="text-burgundy underline underline-offset-2">Vectorscope</a>{" · "}
                      <a href="https://github.com/vector-lab-tools/manifoldscope" target="_blank" rel="noopener noreferrer" className="text-burgundy underline underline-offset-2">Manifoldscope</a>{" · "}
                      <a href="https://github.com/vector-lab-tools/theoryscope" target="_blank" rel="noopener noreferrer" className="text-burgundy underline underline-offset-2">Theoryscope</a>
                    </p>
                  </div>
                </div>
              </div>

              <div className="px-6 py-4 border-t border-parchment flex items-center gap-4 font-sans text-body-sm">
                <a
                  href="https://github.com/vector-lab-tools/diffusionatlas"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-burgundy underline underline-offset-2 hover:text-burgundy-900"
                >
                  GitHub
                </a>
                <a
                  href="https://stunlaw.blogspot.com/2026/02/vector-theory.html"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-burgundy underline underline-offset-2 hover:text-burgundy-900"
                >
                  Vector Theory
                </a>
                <a
                  href="https://stunlaw.blogspot.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-burgundy underline underline-offset-2 hover:text-burgundy-900"
                >
                  Stunlaw
                </a>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
}
