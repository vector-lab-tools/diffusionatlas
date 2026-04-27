/**
 * GenEval-lite task pack for Compositional Bench.
 *
 * Minimal v0.1 set: 4 categories × 3 prompts each = 12 tasks. Each task
 * carries an expected criterion that scoring eventually checks. For now
 * the user marks pass/fail manually; CLIP-based auto-scoring is v0.2+.
 *
 * References:
 *   GenEval: https://arxiv.org/abs/2310.11513
 *   T2I-CompBench: https://arxiv.org/abs/2307.06350
 */

export type TaskCategoryId =
  | "single-object"
  | "two-objects"
  | "counting"
  | "colour-binding";

export interface BenchTask {
  id: string;
  category: TaskCategoryId;
  prompt: string;
  /** Plain-language criterion shown to the user when scoring. */
  criterion: string;
}

export interface TaskCategory {
  id: TaskCategoryId;
  label: string;
  description: string;
}

export const CATEGORIES: TaskCategory[] = [
  {
    id: "single-object",
    label: "Single object",
    description: "Renders one named object cleanly.",
  },
  {
    id: "two-objects",
    label: "Two objects",
    description: "Renders both named objects, distinct and identifiable.",
  },
  {
    id: "counting",
    label: "Counting",
    description: "Renders the correct number of the named object.",
  },
  {
    id: "colour-binding",
    label: "Colour binding",
    description: "Renders each object in its specified colour.",
  },
];

export const TASKS: BenchTask[] = [
  // single-object
  { id: "so-1", category: "single-object", prompt: "a photograph of a cat", criterion: "Image contains exactly one cat." },
  { id: "so-2", category: "single-object", prompt: "a photograph of a chair", criterion: "Image contains exactly one chair." },
  { id: "so-3", category: "single-object", prompt: "a photograph of an apple", criterion: "Image contains exactly one apple." },

  // two-objects
  { id: "to-1", category: "two-objects", prompt: "a photograph of a cat and a dog", criterion: "Both a cat and a dog visible." },
  { id: "to-2", category: "two-objects", prompt: "a photograph of a book and a cup", criterion: "Both a book and a cup visible." },
  { id: "to-3", category: "two-objects", prompt: "a photograph of a chair and a table", criterion: "Both a chair and a table visible." },

  // counting
  { id: "co-1", category: "counting", prompt: "a photograph of three apples on a table", criterion: "Exactly three apples." },
  { id: "co-2", category: "counting", prompt: "a photograph of two birds on a branch", criterion: "Exactly two birds." },
  { id: "co-3", category: "counting", prompt: "a photograph of four candles", criterion: "Exactly four candles." },

  // colour-binding
  { id: "cb-1", category: "colour-binding", prompt: "a photograph of a red cube next to a blue sphere", criterion: "Red cube and blue sphere, colours not swapped." },
  { id: "cb-2", category: "colour-binding", prompt: "a photograph of a yellow car and a green bicycle", criterion: "Yellow car and green bicycle, colours not swapped." },
  { id: "cb-3", category: "colour-binding", prompt: "a photograph of a white mug on a black saucer", criterion: "White mug on a black saucer." },
];

export function tasksByCategory(): Record<TaskCategoryId, BenchTask[]> {
  const out: Record<TaskCategoryId, BenchTask[]> = {
    "single-object": [],
    "two-objects": [],
    counting: [],
    "colour-binding": [],
  };
  for (const t of TASKS) out[t.category].push(t);
  return out;
}
