"use client";

import { type ReactNode } from "react";
import { lookup } from "@/lib/docs/glossary";

interface TableProps {
  headers: string[];
  rows: ReactNode[][];
  /** Right-align numeric columns (zero-indexed). */
  numericColumns?: number[];
  caption?: string;
  /** Override or supplement glossary lookups for column descriptions. */
  descriptions?: Record<string, string>;
}

/**
 * Editorial-styled quantitative table for Deep Dive panels. Zebra rows,
 * compact font, right-aligned numerics. Pure presentation — no sorting.
 */
export function Table({ headers, rows, numericColumns, caption, descriptions }: TableProps) {
  const isNumeric = (col: number) => numericColumns?.includes(col) ?? false;
  const describe = (h: string) => descriptions?.[h] ?? lookup(h);
  return (
    <div className="overflow-x-auto">
      <table className="w-full font-sans text-caption border-collapse">
        {caption && <caption className="text-left text-muted-foreground mb-2">{caption}</caption>}
        <thead>
          <tr className="border-b border-parchment-dark">
            {headers.map((h, i) => {
              const desc = describe(h);
              return (
                <th
                  key={i}
                  title={desc}
                  className={`px-2 py-1.5 font-semibold uppercase tracking-wider text-muted-foreground ${
                    isNumeric(i) ? "text-right" : "text-left"
                  } ${desc ? "cursor-help underline decoration-dotted decoration-muted-foreground/40 underline-offset-4" : ""}`}
                >
                  {h}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rIdx) => (
            <tr key={rIdx} className={rIdx % 2 === 0 ? "" : "bg-cream/30"}>
              {row.map((cell, cIdx) => (
                <td
                  key={cIdx}
                  className={`px-2 py-1.5 border-b border-parchment ${
                    isNumeric(cIdx) ? "text-right tabular-nums" : "text-left"
                  }`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
