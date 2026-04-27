"use client";

import { Download } from "lucide-react";

interface ExportButtonsProps {
  onCsv?: () => void;
  onPdf?: () => void;
  onJson?: () => void;
  disabled?: boolean;
}

export function ExportButtons({ onCsv, onPdf, onJson, disabled }: ExportButtonsProps) {
  const cls = "btn-editorial-secondary px-2 py-1 text-caption flex items-center gap-1";
  const dis = disabled ? "opacity-40 cursor-not-allowed" : "";
  return (
    <div className="flex items-center gap-2">
      {onCsv && (
        <button onClick={onCsv} disabled={disabled} className={`${cls} ${dis}`} title="Download CSV">
          <Download size={12} /> CSV
        </button>
      )}
      {onPdf && (
        <button onClick={onPdf} disabled={disabled} className={`${cls} ${dis}`} title="Download PDF">
          <Download size={12} /> PDF
        </button>
      )}
      {onJson && (
        <button onClick={onJson} disabled={disabled} className={`${cls} ${dis}`} title="Download JSON">
          <Download size={12} /> JSON
        </button>
      )}
    </div>
  );
}
