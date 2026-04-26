/**
 * Diffusion Atlas
 * Concept and Design: David M. Berry, University of Sussex
 * Implemented with Claude Code
 */

import type { Metadata } from "next";
import { Libre_Baskerville, Source_Serif_4, Inter } from "next/font/google";
import "./globals.css";

const libreBaskerville = Libre_Baskerville({
  subsets: ["latin"],
  weight: ["400", "700"],
  style: ["normal", "italic"],
  variable: "--font-libre-baskerville",
  display: "swap",
});

const sourceSerif = Source_Serif_4({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  style: ["normal", "italic"],
  variable: "--font-source-serif",
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Diffusion Atlas",
  description: "Manifold geometry and benchmarking for diffusion models. Atlas + Bench in one instrument.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${libreBaskerville.variable} ${sourceSerif.variable} ${inter.variable}`}
      suppressHydrationWarning
    >
      <body className="font-body antialiased bg-ivory text-ink selection:bg-burgundy/20 selection:text-burgundy-900">
        {children}
      </body>
    </html>
  );
}
