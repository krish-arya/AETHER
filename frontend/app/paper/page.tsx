import type { Metadata } from "next";

import { PlannedModule } from "@/components/ui/planned-module";

export const metadata: Metadata = { title: "Paper Trading" };

export default function PaperPage() {
  return (
    <PlannedModule
      title="Paper Trading"
      phase={7}
      phaseName="Dashboard and Paper Trading"
      summary="A simulated portfolio that follows the strategy forward in time with virtual cash. Paper trading never sends an order to a real broker — no broker integration exists in this platform."
      willShow={[
        "Simulated portfolio",
        "Virtual cash balance",
        "Orders",
        "Executed trades",
        "Rejected trades and reasons",
        "Transaction costs",
        "Daily P&L",
        "Strategy performance",
      ]}
      dependsOn={[
        "Portfolio simulator and cost model (Phase 2)",
        "Risk engine",
        "Routine market-data ingestion",
      ]}
    />
  );
}
