import type { Metadata } from "next";
import { Suspense } from "react";

import { ExperimentsView } from "@/components/experiments/experiments-view";
import { Panel, PanelSkeleton } from "@/components/ui/panel";

export const metadata: Metadata = { title: "Experiment Tracking" };

export default function Page() {
  return (
    <div className="space-y-4">
      <Suspense fallback={<PanelSkeleton title="Experiment Tracking" />}>
        <ExperimentsView />
      </Suspense>

      <Panel title="Not built yet on this page">
        <p className="max-w-3xl text-sm leading-6 text-muted">
          Loss curves, hyper-parameter comparison across runs and per-fold drill-down. MLflow is not used (it depends on a library this machine&apos;s security policy blocks); the registry is plain JSON model cards under artifacts/models.
        </p>
      </Panel>
    </div>
  );
}
