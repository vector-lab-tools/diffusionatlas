"use client";

import { useEffect, useRef } from "react";

// Open overlays in mount order; Escape closes only the topmost, so a modal
// opened from inside another closes first.
const stack: symbol[] = [];

export function useEscape(active: boolean, onEscape: () => void) {
  const handler = useRef(onEscape);
  handler.current = onEscape;

  useEffect(() => {
    if (!active) return;
    const id = Symbol("overlay");
    stack.push(id);
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || stack[stack.length - 1] !== id) return;
      e.stopImmediatePropagation();
      handler.current();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      const i = stack.indexOf(id);
      if (i !== -1) stack.splice(i, 1);
    };
  }, [active]);
}
