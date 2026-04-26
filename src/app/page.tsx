/**
 * Diffusion Atlas
 * Manifold geometry and benchmarking for diffusion models.
 *
 * Concept and Design: David M. Berry
 * University of Sussex
 * https://stunlaw.blogspot.com
 *
 * Implemented with Claude Code
 * MIT Licence
 */

"use client";

import { useState } from "react";
import { Providers } from "./providers";
import { Header } from "@/components/layout/Header";
import { TabNav, type TabId } from "@/components/layout/TabNav";
import { StatusBar } from "@/components/layout/StatusBar";
import { SettingsPanel } from "@/components/layout/SettingsPanel";
import { DenoiseTrajectory } from "@/components/operations/DenoiseTrajectory";
import { GuidanceSweep } from "@/components/operations/GuidanceSweep";
import { LatentNeighbourhood } from "@/components/operations/LatentNeighbourhood";
import { CompositionalBench } from "@/components/operations/CompositionalBench";
import { LibraryBrowse } from "@/components/operations/LibraryBrowse";

function AppContent() {
  const [activeTab, setActiveTab] = useState<TabId>("trajectory");
  const [lastQueryTime] = useState<number | undefined>(undefined);

  return (
    <div className="min-h-screen flex flex-col">
      <Header activeTab={activeTab} />
      <TabNav activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="flex-1 px-6 py-6 max-w-6xl mx-auto w-full">
        {activeTab === "trajectory" && <DenoiseTrajectory />}
        {activeTab === "sweep" && <GuidanceSweep />}
        {activeTab === "neighbourhood" && <LatentNeighbourhood />}
        {activeTab === "compositional" && <CompositionalBench />}
        {activeTab === "library" && <LibraryBrowse />}
      </main>

      <StatusBar lastQueryTime={lastQueryTime} />
      <SettingsPanel />
    </div>
  );
}

export default function Home() {
  return (
    <Providers>
      <AppContent />
    </Providers>
  );
}
