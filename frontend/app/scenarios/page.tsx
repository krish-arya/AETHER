import type { Metadata } from "next";

import { PlannedModule } from "@/components/ui/planned-module";

export const metadata: Metadata = { title: "GAN Scenario Explorer" };

export default function ScenariosPage() {
  return (
    <PlannedModule
      title="GAN Scenario Explorer"
      phase={4}
      phaseName="GAN Engine"
      summary="Regime-conditioned synthetic market scenarios from a TimeGAN baseline, evaluated statistically against history and checked for memorisation. Scenarios are for stress testing and robustness — never a price forecast."
      willShow={[
        "Historical vs synthetic return distributions",
        "Synthetic price trajectories",
        "Volatility comparisons",
        "Cross-asset correlation matrices",
        "Stress scenarios",
        "Regime-specific distributions",
        "Evaluation metrics per model run",
        "Clear labelling of all synthetic data",
      ]}
      dependsOn={[
        "Validated real historical dataset (Phase 1)",
        "Market regime classifier (Phase 2)",
        "PyTorch training pipeline and model registry",
      ]}
    />
  );
}
