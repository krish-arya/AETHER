import type { Metadata } from "next";

import { PlannedModule } from "@/components/ui/planned-module";

export const metadata: Metadata = { title: "Experiment Tracking" };

export default function ExperimentsPage() {
  return (
    <PlannedModule
      title="Experiment Tracking"
      phase={5}
      phaseName="RL Portfolio Optimization (model registry)"
      summary="Every training run with its configuration, dataset version, seeds and metrics, so results can be reproduced and compared — including the runs that did not work."
      willShow={[
        "Model versions",
        "Training runs",
        "Hyperparameters",
        "Dataset versions used",
        "Performance metrics",
        "Training duration",
        "Run-to-run comparisons",
        "Results across random seeds",
      ]}
      dependsOn={[
        "Dataset versioning (Phase 1 — done)",
        "MLflow tracking server",
        "Model registry tables",
      ]}
    />
  );
}
