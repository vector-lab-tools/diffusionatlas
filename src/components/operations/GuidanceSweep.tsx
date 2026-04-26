"use client";

import { OperationStub } from "@/components/shared/OperationStub";

export function GuidanceSweep() {
  return (
    <OperationStub
      title="Guidance Sweep"
      description="Generate the same prompt and seed across a range of CFG values. Reveals the controllability surface and where mode collapse begins."
      backendNote="Hosted-capable. All providers supported."
    />
  );
}
