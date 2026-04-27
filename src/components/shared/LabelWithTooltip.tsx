"use client";

import { type ReactNode } from "react";
import { lookup } from "@/lib/docs/glossary";

interface LabelWithTooltipProps {
  /** Text shown as the form-field label. Looked up against the glossary. */
  label: string;
  /** Override or supplement the glossary definition. */
  description?: string;
  children: ReactNode;
  className?: string;
}

/**
 * Form field label with a hover tooltip drawn from the shared glossary.
 * Uses the native browser `title` attribute — slightly slow but zero-dep
 * and accessible. The label gets a dotted underline so the hover affordance
 * is visible.
 */
export function LabelWithTooltip({ label, description, children, className }: LabelWithTooltipProps) {
  const tip = description ?? lookup(label);
  return (
    <label className={className ?? "block"} title={tip}>
      <span
        className={`font-sans text-caption uppercase tracking-wider text-muted-foreground ${
          tip ? "cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4" : ""
        }`}
      >
        {label}
      </span>
      {children}
    </label>
  );
}
