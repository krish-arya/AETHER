import type { Metadata } from "next";
import { Suspense } from "react";

import { PaperTrading } from "@/components/paper/paper-trading";
import { Panel, PanelSkeleton } from "@/components/ui/panel";

export const metadata: Metadata = { title: "Paper Trading" };

export default function Page() {
  return (
    <div className="space-y-4">
      <Suspense fallback={<PanelSkeleton title="Paper Trading" />}>
        <PaperTrading />
      </Suspense>

      <Panel title="How paper trading works here">
        <p className="max-w-3xl text-sm leading-6 text-muted">
          A paper account is replayed through the same simulator, costs and risk limits as the backtests, from its start date to the latest session in the loaded data. Orders are filled at the next session&apos;s open, so an order placed on the latest session stays pending until newer data is loaded. There is no broker integration.
        </p>
      </Panel>
    </div>
  );
}
