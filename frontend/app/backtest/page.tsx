import type { Metadata } from "next";
import { Suspense } from "react";

import { BacktestLab } from "@/components/backtest/backtest-lab";
import { Panel, PanelSkeleton } from "@/components/ui/panel";

export const metadata: Metadata = { title: "Backtesting Lab" };

export default function BacktestPage() {
  return (
    <div className="space-y-4">
      <Suspense fallback={<PanelSkeleton title="Backtesting Lab" />}>
        <BacktestLab />
      </Suspense>

      <Panel title="Not built yet on this page">
        <p className="max-w-3xl text-sm leading-6 text-muted">
          Equity and drawdown charts, choosing assets and date ranges, editing strategy parameters
          and the cost schedule, and comparing several runs side by side are not in this page yet
          (Phase 7). The API already accepts all of those options; RL strategies arrive with
          Phase 5. The strategies here are rule-based baselines, not AI recommendations.
        </p>
      </Panel>
    </div>
  );
}
