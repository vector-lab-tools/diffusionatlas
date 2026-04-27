/**
 * Preset task packs for Compositional Bench, grouped thematically.
 *
 * The "default" pack is GenEval-lite (4 categories × 3 prompts). "smoke" is
 * the fastest possible run (one prompt per category) for development. The
 * deep packs probe one category in depth. The themed packs add prompts the
 * default doesn't cover — spatial relations, negation, and the critical
 * prompts that distinguish Diffusion Atlas from a generic GenEval clone.
 *
 * Users can edit the active pack inline; edits persist per-pack in
 * localStorage so they survive a reload.
 */

import type { BenchTask, TaskCategoryId } from "./tasks";

export interface BenchPack {
  id: string;
  label: string;
  description: string;
  tasks: BenchTask[];
}

// Helper: type-safe BenchTask construction.
function t(id: string, category: TaskCategoryId, prompt: string, criterion: string): BenchTask {
  return { id, category, prompt, criterion };
}

const DEFAULT_TASKS: BenchTask[] = [
  // single-object
  t("so-1", "single-object", "a photograph of a cat", "Image contains exactly one cat."),
  t("so-2", "single-object", "a photograph of a chair", "Image contains exactly one chair."),
  t("so-3", "single-object", "a photograph of an apple", "Image contains exactly one apple."),
  // two-objects
  t("to-1", "two-objects", "a photograph of a cat and a dog", "Both a cat and a dog visible."),
  t("to-2", "two-objects", "a photograph of a book and a cup", "Both a book and a cup visible."),
  t("to-3", "two-objects", "a photograph of a chair and a table", "Both a chair and a table visible."),
  // counting
  t("co-1", "counting", "a photograph of three apples on a table", "Exactly three apples."),
  t("co-2", "counting", "a photograph of two birds on a branch", "Exactly two birds."),
  t("co-3", "counting", "a photograph of four candles", "Exactly four candles."),
  // colour-binding
  t("cb-1", "colour-binding", "a photograph of a red cube next to a blue sphere", "Red cube and blue sphere, colours not swapped."),
  t("cb-2", "colour-binding", "a photograph of a yellow car and a green bicycle", "Yellow car and green bicycle, colours not swapped."),
  t("cb-3", "colour-binding", "a photograph of a white mug on a black saucer", "White mug on a black saucer."),
];

const SMOKE_TASKS: BenchTask[] = [
  t("smoke-so", "single-object", "a photograph of a cat", "One cat."),
  t("smoke-to", "two-objects", "a photograph of a cat and a dog", "Cat and dog."),
  t("smoke-co", "counting", "a photograph of three apples", "Three apples."),
  t("smoke-cb", "colour-binding", "a photograph of a red cube next to a blue sphere", "Red cube, blue sphere."),
];

const SINGLE_OBJECT_DEEP: BenchTask[] = [
  t("sod-1", "single-object", "a photograph of a cat", "One cat."),
  t("sod-2", "single-object", "a photograph of a chair", "One chair."),
  t("sod-3", "single-object", "a photograph of an apple", "One apple."),
  t("sod-4", "single-object", "a photograph of a teapot", "One teapot."),
  t("sod-5", "single-object", "a photograph of a violin", "One violin."),
  t("sod-6", "single-object", "a photograph of a grandfather clock", "One grandfather clock."),
];

const COUNTING_DEEP: BenchTask[] = [
  t("cnd-1", "counting", "a photograph of one apple", "Exactly one apple."),
  t("cnd-2", "counting", "a photograph of two apples", "Exactly two apples."),
  t("cnd-3", "counting", "a photograph of three apples", "Exactly three apples."),
  t("cnd-4", "counting", "a photograph of four apples", "Exactly four apples."),
  t("cnd-5", "counting", "a photograph of five apples on a table", "Exactly five apples."),
  t("cnd-6", "counting", "a photograph of seven apples in a bowl", "Exactly seven apples."),
];

const COLOUR_DEEP: BenchTask[] = [
  t("cld-1", "colour-binding", "a photograph of a red apple and a green pear", "Red apple, green pear, not swapped."),
  t("cld-2", "colour-binding", "a photograph of a blue car and a yellow bus", "Blue car, yellow bus."),
  t("cld-3", "colour-binding", "a photograph of a purple flower in an orange vase", "Purple flower, orange vase."),
  t("cld-4", "colour-binding", "a photograph of a black cat on a white sofa", "Black cat, white sofa."),
  t("cld-5", "colour-binding", "a photograph of a pink umbrella next to a brown briefcase", "Pink umbrella, brown briefcase."),
];

// Spatial relations — borrowed shape from T2I-CompBench.
const SPATIAL: BenchTask[] = [
  t("sp-1", "two-objects", "a photograph of a book on a chair", "Book is on the chair, not next to it."),
  t("sp-2", "two-objects", "a photograph of a cat under a table", "Cat is under the table."),
  t("sp-3", "two-objects", "a photograph of a lamp to the left of a sofa", "Lamp is on the left."),
  t("sp-4", "two-objects", "a photograph of a teapot between two cups", "Teapot in the middle, cup on each side."),
  t("sp-5", "two-objects", "a photograph of a dog in front of a fireplace", "Dog is in front, fireplace behind."),
];

// Negation — diffusion famously struggles with "without" / "no".
const NEGATION: BenchTask[] = [
  t("ng-1", "single-object", "a photograph of a kitchen without any food", "Kitchen with no food visible."),
  t("ng-2", "single-object", "a photograph of a beach with no people", "Beach, no people."),
  t("ng-3", "single-object", "a photograph of a wall with nothing on it", "Plain wall, no decoration."),
  t("ng-4", "two-objects", "a photograph of a forest without trees", "Forest setting but no trees rendered."),
];

// Berry-voiced critical pack: probes what the manifold will and won't render.
// These are the prompts where compositional failure is the *finding*.
const CRITICAL: BenchTask[] = [
  t("cr-1", "single-object", "a photograph of a courtroom without a judge", "Courtroom with no figure on the bench."),
  t("cr-2", "single-object", "a photograph of a hospital ward with no nurses", "Hospital ward, no nursing staff visible."),
  t("cr-3", "single-object", "a photograph of a factory floor with no workers", "Factory interior, no workers."),
  t("cr-4", "two-objects", "a photograph of a march with no protesters", "Street march scene, no marchers."),
  t("cr-5", "single-object", "a photograph of a server farm with no humans", "Data centre, no human figures."),
];

export const BENCH_PACKS: BenchPack[] = [
  { id: "default", label: "Default (GenEval-lite)", description: "Four categories × three prompts each. The standard.", tasks: DEFAULT_TASKS },
  { id: "smoke", label: "Smoke (4)", description: "One prompt per category. Fastest possible run for testing.", tasks: SMOKE_TASKS },
  { id: "single-deep", label: "Single object (6)", description: "Varied single objects to probe basic generative fidelity.", tasks: SINGLE_OBJECT_DEEP },
  { id: "counting-deep", label: "Counting (6)", description: "1–7 of the same object. Where counting collapses is the finding.", tasks: COUNTING_DEEP },
  { id: "colour-deep", label: "Colour binding (5)", description: "Varied colour-object pairs. Diffusion is known to swap.", tasks: COLOUR_DEEP },
  { id: "spatial", label: "Spatial (5)", description: "On, under, left, between, in front. T2I-CompBench-style relations.", tasks: SPATIAL },
  { id: "negation", label: "Negation (4)", description: "‘without’, ‘no’, ‘nothing’. Probes the deficit diffusion shares with LLMs.", tasks: NEGATION },
  { id: "critical", label: "Critical (5)", description: "Probes the manifold's hegemonic defaults: whose absence does the geometry refuse to render?", tasks: CRITICAL },
];

export function packById(id: string): BenchPack | undefined {
  return BENCH_PACKS.find((p) => p.id === id);
}

export const DEFAULT_PACK_ID = "default";
