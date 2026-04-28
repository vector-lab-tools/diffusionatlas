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
import { VERSION } from "@/lib/version";

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

/**
 * Free-form notes section, rendered below the glossary on the Key page.
 * Use for op-specific commentary that doesn't fit the term/definition
 * shape — e.g. why certain CFG cells fail on a particular hardware
 * stack, or how to interpret a drift curve.
 */
export interface PdfNote {
  title: string;
  /** Body. `\n\n` separates paragraphs; `\n` is a soft wrap. Bullet
   * lines starting with "• " or "- " render as visual bullets. */
  body: string;
}

/**
 * A simple per-axis line chart for embedding in the PDF — drawn
 * natively with jspdf line primitives so we don't have to embed an
 * SVG-to-PNG roundtrip. Currently used for the Guidance Sweep drift
 * curve; the shape is general enough that other ops can reuse it.
 */
export interface PdfDriftCurve {
  label: string;
  caption?: string;
  /** Domain values (CFG numbers, seed offsets, …). Same length as `values`. */
  domain: number[];
  /** Range values in [0, 1], or `null` for missing cells. */
  values: Array<number | null>;
  /** X-axis label. Defaults to the chart label. */
  xAxisLabel?: string;
  /** Y-axis label. Defaults to "drift". */
  yAxisLabel?: string;
}

/**
 * One self-contained block within the PDF — a layer / lane / sweep entry.
 * Images and tables stay together rather than being lumped across the
 * whole document. Newest-first ordering is the caller's responsibility.
 */
export interface PdfGroup {
  title: string;
  caption?: string;
  images?: PdfImage[];
  tables?: PdfAppendixSection[];
}

export interface PdfDoc {
  meta: PdfMeta;
  /** Optional summary table on the main page (kept short — high-level only). */
  table?: PdfTable;
  images?: PdfImage[];
  /** Detailed deep-dive data; rendered on a new page (or pages) at the end. */
  appendix?: PdfAppendixSection[];
  /**
   * Per-group layout: each group renders its own header, image grid, and
   * tables on its own page(s). Used by Denoise Trajectory so each layer's
   * camera roll, latent-geometry table, and image-stats table stay
   * together. When provided, takes precedence over flat `images` / `appendix`.
   */
  groups?: PdfGroup[];
  /** Definitions for the parameters and column headers used in this doc. */
  glossary?: PdfGlossaryEntry[];
  /** Op-specific commentary rendered after the glossary on the Key page. */
  notes?: PdfNote[];
  /** Drift / line-chart visualisations rendered on their own page block. */
  driftCurves?: PdfDriftCurve[];
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
  // Branding band — small all-caps "DIFFUSION ATLAS" wordmark in
  // burgundy at the top-left, version + URL at the top-right, hairline
  // rule beneath. Mirrors the on-screen Vector Lab editorial style.
  doc.setFont("helvetica", "bold");
  doc.setFontSize(8);
  doc.setTextColor(124, 45, 54);
  doc.setCharSpace(2);
  doc.text("DIFFUSION ATLAS", MARGIN, MARGIN);
  doc.setCharSpace(0);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7);
  doc.setTextColor(140);
  doc.text(
    `v${VERSION} · vector-lab-tools.github.io`,
    PAGE_W - MARGIN,
    MARGIN,
    { align: "right" },
  );
  doc.setDrawColor(200);
  doc.setLineWidth(0.2);
  doc.line(MARGIN, MARGIN + 2, PAGE_W - MARGIN, MARGIN + 2);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.setTextColor(124, 45, 54);
  doc.text(meta.title, MARGIN, MARGIN + 12);

  let y = MARGIN + 20;
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

function drawImages(doc: jsPDF, startY: number, images: PdfImage[]): number {
  // 3-up grid, each ~58mm wide. Returns the y-cursor after drawing so a
  // caller can continue rendering tables underneath.
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
  // Advance past the trailing partial row, if any.
  if (col !== 0) y += cellH + 8;
  return y;
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

function drawGroups(doc: jsPDF, groups: PdfGroup[]): void {
  groups.forEach((group, gi) => {
    // Each group starts on a fresh page so layers do not get visually
    // entangled. The first group can claim its own page right after the
    // header; subsequent groups always break.
    if (gi === 0) doc.addPage(); else doc.addPage();
    let y = MARGIN + 4;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.setTextColor(124, 45, 54);
    doc.text(group.title, MARGIN, y);
    y += 5;

    if (group.caption) {
      doc.setFont("helvetica", "italic");
      doc.setFontSize(8);
      doc.setTextColor(110);
      const lines = doc.splitTextToSize(group.caption, PAGE_W - 2 * MARGIN);
      doc.text(lines, MARGIN, y);
      y += lines.length * 3.5 + 2;
    }

    doc.setDrawColor(220);
    doc.setLineWidth(0.2);
    doc.line(MARGIN, y, PAGE_W - MARGIN, y);
    y += 4;

    if (group.images && group.images.length > 0) {
      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.setTextColor(60);
      doc.text("Samples", MARGIN, y);
      y = drawImages(doc, y + 2, group.images);
    }

    if (group.tables && group.tables.length > 0) {
      for (const section of group.tables) {
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
  });
}

function drawGlossary(doc: jsPDF, entries: PdfGlossaryEntry[], notes?: PdfNote[]): number {
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

  // Free-form notes below the glossary. Rendered with a small heading
  // per note, then the body — paragraph breaks on `\n\n`, soft wraps
  // inside paragraphs respected.
  if (notes && notes.length > 0) {
    y += 4;
    doc.setDrawColor(220);
    doc.line(MARGIN, y, PAGE_W - MARGIN, y);
    y += 5;
    for (const note of notes) {
      y = ensureSpace(doc, y, 14);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.setTextColor(124, 45, 54);
      doc.text(note.title, MARGIN, y);
      y += 5;

      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.setTextColor(80);
      const paragraphs = note.body.split(/\n\n+/);
      for (const para of paragraphs) {
        const lines = doc.splitTextToSize(para, PAGE_W - 2 * MARGIN);
        y = ensureSpace(doc, y, lines.length * 3.5 + 2);
        doc.text(lines, MARGIN, y);
        y += lines.length * 3.5 + 3;
      }
      y += 2;
    }
  }
  return y;
}

/**
 * Draw the drift curves on a fresh page. Native jspdf line primitives
 * — no SVG-to-PNG roundtrip. Mirrors the on-screen DriftCurve: dashed
 * gridlines at 0 / 0.25 / 0.5 / 0.75 / 1.0 on Y, dashed verticals at
 * each domain sample, solid axes, burgundy line + point markers, value
 * labels above each point.
 */
function drawDriftCurves(doc: jsPDF, curves: PdfDriftCurve[]): void {
  if (curves.length === 0) return;
  doc.addPage();
  let y = MARGIN + 4;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  doc.setTextColor(124, 45, 54);
  doc.text("Drift curves", MARGIN, y);
  y += 5;

  doc.setFont("helvetica", "italic");
  doc.setFontSize(8);
  doc.setTextColor(110);
  const intro =
    "Perceptual hash distance from the cell nearest CFG ≈ 7.5 (the conventional baseline). " +
    "0 = visually identical to the baseline, 1 = maximally different. The curve traces the " +
    "controllability surface — where it bends sharply, the model's behaviour at that CFG is " +
    "qualitatively different from the default.";
  const introLines = doc.splitTextToSize(intro, PAGE_W - 2 * MARGIN);
  doc.text(introLines, MARGIN, y);
  y += introLines.length * 3.5 + 4;

  for (const c of curves) {
    const chartW = PAGE_W - 2 * MARGIN;
    const chartH = 60;
    const padL = 16;
    const padR = 8;
    const padT = 8;
    const padB = 12;
    const x0 = MARGIN + padL;
    const x1 = MARGIN + chartW - padR;
    const yBase = y + chartH - padB;
    const yTop = y + padT;

    y = ensureSpace(doc, y, chartH + 12);

    // Title.
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9);
    doc.setTextColor(60);
    doc.text(c.label, MARGIN, y - 1);

    const minX = Math.min(...c.domain);
    const maxX = Math.max(...c.domain);
    const xScale = (x: number) => x0 + ((x - minX) / Math.max(1, maxX - minX)) * (x1 - x0);
    const yScale = (v: number) => yBase - v * (yBase - yTop);

    // Horizontal gridlines + Y-tick labels.
    const yTicks = [0, 0.25, 0.5, 0.75, 1.0];
    doc.setLineWidth(0.1);
    doc.setFontSize(6);
    doc.setTextColor(140);
    for (const t of yTicks) {
      const py = yScale(t);
      if (t === 0 || t === 1) {
        doc.setDrawColor(180);
        doc.setLineDashPattern([], 0);
      } else {
        doc.setDrawColor(220);
        doc.setLineDashPattern([0.5, 0.5], 0);
      }
      doc.line(x0, py, x1, py);
      doc.text(t.toFixed(t === 0 || t === 1 ? 0 : 2), x0 - 1.5, py + 1, { align: "right" });
    }

    // Vertical gridlines at each domain value + X-tick labels.
    doc.setDrawColor(220);
    doc.setLineDashPattern([0.5, 0.5], 0);
    for (const dx of c.domain) {
      const px = xScale(dx);
      doc.line(px, yTop, px, yBase);
      doc.text(String(dx), px, yBase + 4, { align: "center" });
    }

    // Solid axes on top.
    doc.setLineDashPattern([], 0);
    doc.setDrawColor(140);
    doc.setLineWidth(0.3);
    doc.line(x0, yBase, x1, yBase);
    doc.line(x0, yTop, x0, yBase);

    // Burgundy line connecting non-null points.
    doc.setDrawColor(124, 45, 54);
    doc.setLineWidth(0.6);
    let prevX: number | null = null;
    let prevY: number | null = null;
    for (let i = 0; i < c.domain.length; i++) {
      const v = c.values[i];
      if (v == null) continue;
      const px = xScale(c.domain[i]);
      const py = yScale(v);
      if (prevX != null && prevY != null) {
        doc.line(prevX, prevY, px, py);
      }
      prevX = px;
      prevY = py;
    }

    // Point markers + value labels.
    doc.setFillColor(124, 45, 54);
    doc.setFontSize(6);
    doc.setTextColor(60);
    for (let i = 0; i < c.domain.length; i++) {
      const v = c.values[i];
      if (v == null) continue;
      const px = xScale(c.domain[i]);
      const py = yScale(v);
      doc.circle(px, py, 0.7, "F");
      doc.text(v.toFixed(2), px, py - 1.5, { align: "center" });
    }

    // Axis legends.
    doc.setFontSize(6);
    doc.setTextColor(140);
    doc.text("CFG", (x0 + x1) / 2, yBase + 8, { align: "center" });
    doc.text("drift", x0 - 6, (yTop + yBase) / 2, {
      align: "center",
      angle: 90,
    });

    y += chartH + 6;
  }
}

/**
 * Stamp a Diffusion Atlas footer on every page, called once after the
 * doc is fully built so the page count is final. Burgundy hairline rule
 * + small wordmark on the left and page X/Y + URL on the right.
 */
function stampFooters(doc: jsPDF): void {
  const total = doc.getNumberOfPages();
  const stamped = new Date().toISOString().slice(0, 10);
  for (let i = 1; i <= total; i++) {
    doc.setPage(i);
    const y = PAGE_H - 8;
    doc.setDrawColor(220);
    doc.setLineWidth(0.2);
    doc.line(MARGIN, y - 2, PAGE_W - MARGIN, y - 2);

    doc.setFont("helvetica", "bold");
    doc.setFontSize(7);
    doc.setTextColor(124, 45, 54);
    doc.setCharSpace(2);
    doc.text("DIFFUSION ATLAS", MARGIN, y + 2);
    doc.setCharSpace(0);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(7);
    doc.setTextColor(140);
    doc.text(
      `Generated ${stamped} · v${VERSION} · vector-lab-tools.github.io · page ${i}/${total}`,
      PAGE_W - MARGIN,
      y + 2,
      { align: "right" },
    );
  }
}

export function buildPdf(payload: PdfDoc): jsPDF {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  let y = drawHeader(doc, payload.meta);
  if (payload.table) y = drawTable(doc, y, payload.table);
  if (payload.groups && payload.groups.length > 0) {
    drawGroups(doc, payload.groups);
  } else {
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
  }
  if (payload.driftCurves && payload.driftCurves.length > 0) {
    drawDriftCurves(doc, payload.driftCurves);
  }
  if ((payload.glossary && payload.glossary.length > 0) || (payload.notes && payload.notes.length > 0)) {
    drawGlossary(doc, payload.glossary ?? [], payload.notes);
  }
  stampFooters(doc);
  return doc;
}

export function downloadPdf(filename: string, payload: PdfDoc): void {
  const doc = buildPdf(payload);
  doc.save(filename);
}
