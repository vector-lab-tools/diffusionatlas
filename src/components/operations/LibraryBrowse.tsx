"use client";

import { OperationStub } from "@/components/shared/OperationStub";

export function LibraryBrowse() {
  return (
    <OperationStub
      title="Library"
      description="Saved runs, trajectories, sweeps, and bench results. Browse, replay, compare, export."
      backendNote="Local (IndexedDB)."
    />
  );
}
