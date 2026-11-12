import type { Metadata } from "next";
import { Suspense } from "react";

import { MarketExplorer } from "@/components/market/market-explorer";
import { Panel, PanelSkeleton } from "@/components/ui/panel";

export const metadata: Metadata = { title: "Market Intelligence" };

export default function MarketPage() {
  return (
    <div className="space-y-4">
      <Suspense fallback={<PanelSkeleton title="Universe" />}>
        <MarketExplorer />
      </Suspense>

      <Panel title="Not built yet on this page">
        <p className="max-w-3xl text-sm leading-6 text-muted">
          Index overview, price and indicator charts, sector performance and the market-regime
          timeline are not implemented. Regime detection arrives with Phase 2; charts with the
          Phase 7 dashboard work. This page currently shows the validated universe, the latest bar
          and the latest technical features exactly as stored in the loaded dataset.
        </p>
      </Panel>
    </div>
  );
}
