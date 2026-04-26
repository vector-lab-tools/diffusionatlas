"use client";

import { OperationStub } from "@/components/shared/OperationStub";

export function CompositionalBench() {
  return (
    <OperationStub
      title="Compositional Bench"
      description="GenEval-style scored tasks: single object, two objects, counting, colour, position, colour-attribution. Aggregates to per-category accuracy with a leaderboard."
      backendNote="Hosted-capable. Scoring runs locally via CLIP + heuristics where available."
    />
  );
}
