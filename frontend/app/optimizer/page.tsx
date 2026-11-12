import type { Metadata } from "next";

import { PlannedModule } from "@/components/ui/planned-module";

export const metadata: Metadata = { title: "RL Portfolio Optimizer" };

export default function OptimizerPage() {
  return (
    <PlannedModule
      title="RL Portfolio Optimizer"
      phase={5}
      phaseName="RL Portfolio Optimization"
      summary="Target allocations from a reinforcement-learning agent (PPO first, SAC as an alternative) trained in a Gymnasium portfolio environment. Every allocation passes through deterministic risk controls before it is shown."
      willShow={[
        "Current portfolio",
        "Recommended target allocation",
        "Cash allocation",
        "Asset-level weights",
        "Portfolio risk",
        "Rebalancing recommendations",
        "Comparison with the previous allocation",
        "Evidence behind each allocation",
      ]}
      dependsOn={[
        "Backtesting engine and cost model (Phase 2)",
        "Trading environment and trained agent",
        "Independent risk engine",
      ]}
    />
  );
}
