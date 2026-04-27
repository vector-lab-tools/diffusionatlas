/**
 * Shared PDF export for Deep Dive panels.
 *
 * Builds a research-oriented PDF: title page (operation name, run metadata,
 * model + provider), then a quantitative table, then optional embedded
 * images (gallery cards, trajectory thumbnails, the final image).
 *
 * Uses jspdf + jspdf-autotable, both already installed; no new deps.
 */

import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

export interface PdfMeta {
  /** Operation name in the page header. */
  title: string;
  /** Provider + model line under the title. */
  subtitle?: string;
  /** Free-form metadata rows shown above the table. */
  fields?: Array<{ label: string; value: string | number }>;
}

export interface PdfTable {
  headers: string[];
  rows: Array<Array<string | number>>;
}

export interface PdfImage {
  /** Data URL (data:image/png;base64,…). */
  dataUrl: string;
  caption?: string;
}

export interface PdfDoc {
  meta: PdfMeta;
  table?: PdfTable;
  images?: PdfImage[];
}

const MARGIN = 14;
const PAGE_W = 210; // A4 portrait
const PAGE_H = 297;

function ensureSpace(doc: jsPDF, y: number, needed: number): number {
  if (y + needed > PAGE_H - MARGIN) {
    doc.addPage();
    return MARGIN;
  }
  return y;
}

function drawHeader(doc: jsPDF, meta: PdfMeta): number {
  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.setTextColor(124, 45, 54); // burgundy
  doc.text(meta.title, MARGIN, MARGIN + 6);

  let y = MARGIN + 14;
  if (meta.subtitle) {
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(80);
    doc.text(meta.subtitle, MARGIN, y);
    y += 6;
  }

  if (meta.fields && meta.fields.length > 0) {
    doc.setFontSize(9);
    doc.setTextColor(60);
    for (const f of meta.fields) {
      doc.setFont("helvetica", "bold");
      doc.text(`${f.label}:`, MARGIN, y);
      doc.setFont("helvetica", "normal");
      doc.text(String(f.value), MARGIN + 32, y);
      y += 4.5;
    }
    y += 2;
  }

  // Thin rule under header
  doc.setDrawColor(220);
  doc.setLineWidth(0.2);
  doc.line(MARGIN, y, PAGE_W - MARGIN, y);
  return y + 4;
}

function drawTable(doc: jsPDF, startY: number, table: PdfTable): number {
  autoTable(doc, {
    startY,
    head: [table.headers],
    body: table.rows.map((r) => r.map((c) => String(c))),
    margin: { left: MARGIN, right: MARGIN },
    styles: { fontSize: 8, cellPadding: 1.5 },
    headStyles: { fillColor: [124, 45, 54], textColor: 255, fontStyle: "bold" },
    alternateRowStyles: { fillColor: [248, 246, 241] },
    theme: "grid",
  });
  // jspdf-autotable mutates lastAutoTable on the doc.
  const finalY = (doc as unknown as { lastAutoTable?: { finalY: number } }).lastAutoTable?.finalY ?? startY;
  return finalY + 4;
}

function drawImages(doc: jsPDF, startY: number, images: PdfImage[]): void {
  // 3-up grid, each ~58mm wide.
  const cols = 3;
  const cellW = (PAGE_W - 2 * MARGIN - (cols - 1) * 4) / cols;
  const cellH = cellW; // square
  let y = startY + 2;
  let col = 0;
  for (const img of images) {
    y = ensureSpace(doc, y, cellH + 8);
    const x = MARGIN + col * (cellW + 4);
    try {
      doc.addImage(img.dataUrl, "PNG", x, y, cellW, cellH);
    } catch {
      // Fallback: skip an unrenderable image rather than abort the whole PDF.
    }
    if (img.caption) {
      doc.setFont("helvetica", "normal");
      doc.setFontSize(7);
      doc.setTextColor(80);
      doc.text(img.caption, x, y + cellH + 4, { maxWidth: cellW });
    }
    col++;
    if (col >= cols) {
      col = 0;
      y += cellH + 8;
    }
  }
}

export function buildPdf(payload: PdfDoc): jsPDF {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  let y = drawHeader(doc, payload.meta);
  if (payload.table) y = drawTable(doc, y, payload.table);
  if (payload.images && payload.images.length > 0) {
    y = ensureSpace(doc, y, 60);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10);
    doc.setTextColor(60);
    doc.text("Samples", MARGIN, y);
    drawImages(doc, y + 2, payload.images);
  }
  return doc;
}

export function downloadPdf(filename: string, payload: PdfDoc): void {
  const doc = buildPdf(payload);
  doc.save(filename);
}
