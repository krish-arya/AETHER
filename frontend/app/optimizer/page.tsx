import type { Metadata } from "next";
import { Suspense } from "react";

import { Optimizer } from "@/components/optimizer/optimizer";
import { Panel, PanelSkeleton } from "@/components/ui/panel";

export const metadata: Metadata = { title: "RL Portfolio Optimizer" };

export default function Page() {
  return (
    <div className="space-y-4">
      <Suspense fallback={<PanelSkeleton title="RL Portfolio Optimizer" />}>
        <Optimizer />
      </Suspense>

      <Panel title="How to read this page">
        <p className="max-w-3xl text-sm leading-6 text-muted">
          The allocation comes from a reinforcement-learning policy and is always passed through the deterministic risk engine. The explanation is assembled from the model&apos;s actual output; sensitivities are statistical associations, not causes. Whether this kind of model beats simple strategies is what the Experiment Tracking page reports.
        </p>
      </Panel>
    </div>
  );
}
