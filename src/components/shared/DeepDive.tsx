"use client";

import { useState, type ReactNode } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

interface DeepDiveProps {
  title?: string;
  children: ReactNode;
  /** Optional inline action area on the right of the header (e.g. CSV export). */
  actions?: ReactNode;
  defaultOpen?: boolean;
}

/**
 * Collapsible quantitative-detail panel — same role as Manifold Atlas's
 * Deep Dive. Sits at the bottom of every operation; click to expand.
 */
export function DeepDive({ title = "Deep Dive", children, actions, defaultOpen = false }: DeepDiveProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-t border-parchment mt-6">
      <div className="flex items-center justify-between py-3">
        <button
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-2 font-sans text-caption uppercase tracking-wider font-semibold text-muted-foreground hover:text-foreground transition-colors"
        >
          {open ? <ChevronDown size={14} className="text-burgundy" /> : <ChevronRight size={14} />}
          {title}
        </button>
        {actions && open && <div>{actions}</div>}
      </div>
      {open && <div className="pb-4">{children}</div>}
    </div>
  );
}
