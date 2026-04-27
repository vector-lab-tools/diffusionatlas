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

export interface PdfAppendixSection {
  title: string;
  /** Optional small italic caption under the section title. */
  caption?: string;
  table: PdfTable;
}

export interface PdfGlossaryEntry {
  term: string;
  definition: string;
}

export interface PdfDoc {
  meta: PdfMeta;
  /** Optional summary table on the main page (kept short — high-level only). */
  table?: PdfTable;
  images?: PdfImage[];
  /** Detailed deep-dive data; rendered on a new page (or pages) at the end. */
  appendix?: PdfAppendixSection[];
  /** Definitions for the parameters and column headers used in this doc. */
  glossary?: PdfGlossaryEntry[];
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

function drawAppendix(doc: jsPDF, sections: PdfAppendixSection[]): void {
  doc.addPage();
  let y = MARGIN + 4;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(14);
  doc.setTextColor(124, 45, 54);
  doc.text("Appendix · Deep Dive", MARGIN, y);
  y += 8;
  doc.setDrawColor(220);
  doc.setLineWidth(0.2);
  doc.line(MARGIN, y - 2, PAGE_W - MARGIN, y - 2);

  for (const section of sections) {
    y = ensureSpace(doc, y, 18);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(60);
    doc.text(section.title, MARGIN, y);
    y += 4;

    if (section.caption) {
      doc.setFont("helvetica", "italic");
      doc.setFontSize(8);
      doc.setTextColor(110);
      const lines = doc.splitTextToSize(section.caption, PAGE_W - 2 * MARGIN);
      doc.text(lines, MARGIN, y);
      y += lines.length * 3.5 + 1;
    }

    y = drawTable(doc, y, section.table);
    y += 4;
  }
}

function drawGlossary(doc: jsPDF, entries: PdfGlossaryEntry[]): void {
  doc.addPage();
  let y = MARGIN + 4;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(14);
  doc.setTextColor(124, 45, 54);
  doc.text("Key", MARGIN, y);
  y += 4;
  doc.setFont("helvetica", "italic");
  doc.setFontSize(8);
  doc.setTextColor(110);
  doc.text("Glossary of parameters and column headers used in this report.", MARGIN, y + 3);
  y += 8;
  doc.setDrawColor(220);
  doc.setLineWidth(0.2);
  doc.line(MARGIN, y - 2, PAGE_W - MARGIN, y - 2);

  for (const entry of entries) {
    y = ensureSpace(doc, y, 12);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9);
    doc.setTextColor(60);
    doc.text(entry.term, MARGIN, y);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(80);
    const lines = doc.splitTextToSize(entry.definition, PAGE_W - 2 * MARGIN - 30);
    doc.text(lines, MARGIN + 30, y);
    y += Math.max(lines.length * 3.5, 5) + 1;
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
  if (payload.appendix && payload.appendix.length > 0) {
    drawAppendix(doc, payload.appendix);
  }
  if (payload.glossary && payload.glossary.length > 0) {
    drawGlossary(doc, payload.glossary);
  }
  return doc;
}

export function downloadPdf(filename: string, payload: PdfDoc): void {
  const doc = buildPdf(payload);
  doc.save(filename);
}
