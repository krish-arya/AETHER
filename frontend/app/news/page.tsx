import type { Metadata } from "next";

import { PlannedModule } from "@/components/ui/planned-module";

export const metadata: Metadata = { title: "AI News Intelligence" };

export default function NewsPage() {
  return (
    <PlannedModule
      title="AI News Intelligence"
      phase={3}
      phaseName="LLM Intelligence Engine"
      summary="Structured, source-attributed financial events extracted from news and corporate filings by a provider-agnostic LLM pipeline with schema validation and hallucination controls."
      willShow={[
        "Latest financial events",
        "Company-wise sentiment",
        "Event type and severity",
        "Affected sectors",
        "Impact horizon",
        "Source documents for every event",
        "Source-grounded summaries",
        "Filters: company, sector, date, event type",
      ]}
      dependsOn={[
        "A licensed news / filings source",
        "A configured LLM provider and API key",
        "Event schema, entity resolution and retrieval store",
      ]}
    />
  );
}
