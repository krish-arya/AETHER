import { Suspense } from "react";

import { BuildStatusPanel } from "@/components/overview/build-status-panel";
import { DatasetsPanel } from "@/components/overview/datasets-panel";
import { AlertsPanel, PortfolioPanel, RegimePanel } from "@/components/overview/market-panels";
import { SystemStatusPanel } from "@/components/overview/system-status-panel";
import { PanelSkeleton } from "@/components/ui/panel";

export default function OverviewPage() {
  return (
    <div className="grid gap-4 xl:grid-cols-3">
      <div className="space-y-4">
        <Suspense fallback={<PanelSkeleton title="System status" />}>
          <SystemStatusPanel />
        </Suspense>

        <Suspense fallback={<PanelSkeleton title="Paper portfolio" />}>
          <PortfolioPanel />
        </Suspense>
      </div>

      <Suspense fallback={<PanelSkeleton title="Loaded datasets" className="xl:col-span-2" />}>
        <DatasetsPanel className="xl:col-span-2" />
      </Suspense>

      <Suspense fallback={<PanelSkeleton title="Alerts" className="xl:col-span-3" />}>
        <AlertsPanel className="xl:col-span-3" />
      </Suspense>

      <Suspense fallback={<PanelSkeleton title="Market regime" className="xl:col-span-3" />}>
        <RegimePanel className="xl:col-span-3" />
      </Suspense>

      <Suspense
        fallback={<PanelSkeleton title="Platform build status" className="xl:col-span-3" />}
      >
        <BuildStatusPanel className="xl:col-span-3" />
      </Suspense>
    </div>
  );
}
