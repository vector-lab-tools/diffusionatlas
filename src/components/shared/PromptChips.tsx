"use client";

interface PromptChipsProps {
  /** Pretty label shown to the left of the chips. */
  label?: string;
  /** Active prompt — chip is highlighted when its value matches. */
  active?: string;
  /** Array of preset prompts to show. */
  presets: Array<{ label: string; prompt: string; hint?: string }>;
  onPick: (prompt: string) => void;
}

/**
 * Compact preset-prompt chip row. Click a chip to populate the prompt
 * input. Modeled on Manifold Atlas's preset chips — gives users a curated
 * set of starting points covering composition, counting, colour binding,
 * style, and Berry-voiced critical probes.
 */
export function PromptChips({ label, active, presets, onPick }: PromptChipsProps) {
  return (
    <div className="flex items-start gap-2 flex-wrap">
      {label && (
        <span className="font-sans text-caption text-muted-foreground mt-0.5 flex-shrink-0">
          {label}:
        </span>
      )}
      <div className="flex flex-wrap gap-1">
        {presets.map((p) => {
          const isActive = active === p.prompt;
          return (
            <button
              key={p.label}
              onClick={() => onPick(p.prompt)}
              title={p.hint ?? p.prompt}
              className={`px-2 py-0.5 font-sans text-caption rounded-sm border transition-colors ${
                isActive
                  ? "bg-burgundy text-primary-foreground border-burgundy"
                  : "bg-cream/40 text-foreground border-parchment-dark hover:bg-cream/70"
              }`}
            >
              {p.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Curated preset prompts grouped thematically. Used across Trajectory,
 * Sweep, and Neighbourhood so the user has the same vocabulary of
 * starting points wherever they are. Mix of "well-behaved" prompts
 * (good for showing the operation working) and "compositionally
 * difficult" or "Berry-voiced critical" prompts (good for showing
 * what the manifold cannot do).
 */
export const STARTER_PRESETS = [
  // Well-behaved
  { label: "cat on chair", prompt: "a cat sitting on a wooden chair, photorealistic", hint: "Classic SD-friendly prompt; produces a clean composition." },
  { label: "portrait", prompt: "a portrait of an elderly woman, oil painting", hint: "Tests style + subject. Style words pull strongly in latent space." },
  { label: "landscape", prompt: "a serene mountain lake at sunrise, golden hour", hint: "Wide-frame composition; tests the model's landscape priors." },
  { label: "abstract", prompt: "abstract geometric pattern, blue and orange, high contrast", hint: "No subject, only colour and shape. Reveals the model's compositional defaults when there is nothing to depict." },

  // Compositionally hard
  { label: "red+blue cubes", prompt: "a red cube on a blue cube, photorealistic", hint: "Colour binding + spatial relation. Models often swap the colours." },
  { label: "two objects", prompt: "a cat and a dog on a sofa", hint: "Two-object composition. Tests whether the model renders both distinctly." },
  { label: "counting (3)", prompt: "exactly three apples on a wooden table, photograph", hint: "Counting beyond two is where most models break." },
  { label: "yellow car, green bike", prompt: "a yellow car parked next to a green bicycle on a street", hint: "Colour-attribute binding test. Watch for swaps." },

  // Berry-voiced critical probes — what the manifold refuses to render
  { label: "courtroom no judge", prompt: "a courtroom without a judge, photograph", hint: "Critical probe: the manifold tends to fill the bench anyway. The judge is geometrically necessary." },
  { label: "hospital no nurses", prompt: "a hospital ward with no nurses, photograph", hint: "Critical probe: which absences the geometry refuses to render." },
  { label: "factory no workers", prompt: "a factory floor with no workers, photograph", hint: "Critical probe: where the training data's social ontology shows through." },
  { label: "march no protesters", prompt: "a street march with no protesters, photograph", hint: "Critical probe: the geometry's permitted form for a march." },
];
