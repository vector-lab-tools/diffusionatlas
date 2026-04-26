"use client";

import { cn } from "@/lib/utils";
import { Map, Trophy, Library, type LucideIcon } from "lucide-react";

export type TabId =
  | "trajectory"
  | "sweep"
  | "neighbourhood"
  | "compositional"
  | "library";
export type GroupId = "atlas" | "bench" | "library";

interface TabGroup {
  id: GroupId;
  label: string;
  description: string;
  icon: LucideIcon;
  tabs: Array<{ id: TabId; label: string }>;
}

const GROUPS: TabGroup[] = [
  {
    id: "atlas",
    label: "Atlas",
    description: "Latent geometry of the denoising trajectory",
    icon: Map,
    tabs: [
      { id: "trajectory", label: "Denoise Trajectory" },
      { id: "sweep", label: "Guidance Sweep" },
      { id: "neighbourhood", label: "Latent Neighbourhood" },
    ],
  },
  {
    id: "bench",
    label: "Bench",
    description: "Compositional fidelity and scored evaluation",
    icon: Trophy,
    tabs: [
      { id: "compositional", label: "Compositional" },
    ],
  },
  {
    id: "library",
    label: "Library",
    description: "Saved runs and presets",
    icon: Library,
    tabs: [{ id: "library", label: "Browse" }],
  },
];

function getGroup(tabId: TabId): GroupId {
  for (const group of GROUPS) {
    if (group.tabs.some((t) => t.id === tabId)) return group.id;
  }
  return "atlas";
}

export function getGroupLabel(tabId: TabId): string {
  for (const group of GROUPS) {
    if (group.tabs.some((t) => t.id === tabId)) return group.label;
  }
  return "";
}

interface TabNavProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

export function TabNav({ activeTab, onTabChange }: TabNavProps) {
  const activeGroup = getGroup(activeTab);
  const currentGroup = GROUPS.find((g) => g.id === activeGroup)!;

  return (
    <div className="bg-card">
      <div className="px-6 flex gap-0 border-b border-parchment-dark">
        {GROUPS.map((group) => {
          const Icon = group.icon;
          const isActive = group.id === activeGroup;
          return (
            <button
              key={group.id}
              onClick={() => onTabChange(group.tabs[0].id)}
              className={cn(
                "flex items-center gap-2 px-6 py-3 font-sans text-body-sm font-semibold",
                "border-b-[3px] transition-all duration-200",
                isActive
                  ? "border-burgundy text-burgundy bg-background"
                  : "border-transparent text-muted-foreground hover:text-foreground hover:bg-cream/30"
              )}
            >
              <Icon size={16} />
              {group.label}
            </button>
          );
        })}
      </div>

      <div className="px-6 flex gap-1 py-1 border-b border-parchment bg-muted/30">
        {currentGroup.tabs.map((tab) => {
          const isActive = tab.id === activeTab;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={cn(
                "px-4 py-1.5 font-sans text-body-sm font-medium rounded-sm",
                "transition-all duration-200",
                isActive
                  ? "text-primary-foreground bg-burgundy shadow-editorial"
                  : "text-muted-foreground hover:text-foreground hover:bg-cream/50"
              )}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
