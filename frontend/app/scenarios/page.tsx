import type { Metadata } from "next";
import { Suspense } from "react";

import { GanExplorer } from "@/components/scenarios/gan-explorer";
import { Panel, PanelSkeleton } from "@/components/ui/panel";

export const metadata: Metadata = { title: "GAN Scenario Explorer" };

export default function Page() {
  return (
    <div className="space-y-4">
      <Suspense fallback={<PanelSkeleton title="GAN Scenario Explorer" />}>
        <GanExplorer />
      </Suspense>

      <Panel title="Not built yet on this page">
        <p className="max-w-3xl text-sm leading-6 text-muted">
          Charts of synthetic price paths, correlation matrices and regime-specific distributions. Everything generated here is synthetic and labelled so: scenarios are for stress tests and robustness checks, never forecasts.
        </p>
      </Panel>
    </div>
  );
}
