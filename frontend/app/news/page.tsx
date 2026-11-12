import type { Metadata } from "next";
import { Suspense } from "react";

import { NewsFeed } from "@/components/news/news-feed";
import { Panel, PanelSkeleton } from "@/components/ui/panel";

export const metadata: Metadata = { title: "AI News Intelligence" };

export default function Page() {
  return (
    <div className="space-y-4">
      <Suspense fallback={<PanelSkeleton title="AI News Intelligence" />}>
        <NewsFeed />
      </Suspense>

      <Panel title="What this page can and cannot show">
        <p className="max-w-3xl text-sm leading-6 text-muted">
          The news source is recent headlines only (no article bodies, no history), so there is no historical news archive and no news features in backtests or in the RL experiments. Company-level sentiment over time, sector views and source documents beyond the headline need a licensed news feed.
        </p>
      </Panel>
    </div>
  );
}
