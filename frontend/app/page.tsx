import { Suspense } from "react";

import { BuildStatusPanel } from "@/components/overview/build-status-panel";
import { DatasetsPanel } from "@/components/overview/datasets-panel";
import { SystemStatusPanel } from "@/components/overview/system-status-panel";
import { Panel, PanelSkeleton } from "@/components/ui/panel";

export default function OverviewPage() {
  return (
    <div className="grid gap-4 xl:grid-cols-3">
      <div className="space-y-4">
        <Suspense fallback={<PanelSkeleton title="System status" />}>
          <SystemStatusPanel />
        </Suspense>

        <Panel title="Portfolio">
          <p className="text-sm leading-6 text-muted">
            No portfolio exists yet. Portfolio value, returns, drawdown, allocation and risk
            indicators appear here once the backtesting engine (Phase 2) and the RL optimizer
            (Phase 5) produce them. Nothing is shown until then — no placeholder figures.
          </p>
        </Panel>
      </div>

      <Suspense fallback={<PanelSkeleton title="Loaded datasets" className="xl:col-span-2" />}>
        <DatasetsPanel className="xl:col-span-2" />
      </Suspense>

      <Suspense
        fallback={<PanelSkeleton title="Platform build status" className="xl:col-span-3" />}
      >
        <BuildStatusPanel className="xl:col-span-3" />
      </Suspense>
    </div>
  );
}
