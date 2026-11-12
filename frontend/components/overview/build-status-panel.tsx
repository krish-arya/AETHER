"use client";

import { ModuleStatusBadge } from "@/components/ui/badges";
import { Panel } from "@/components/ui/panel";
import { QueryError } from "@/components/ui/query-error";
import { useSystemInfo } from "@/hooks/use-system";

/** Roadmap status straight from the backend's module registry -- not hand-edited here. */
export function BuildStatusPanel({ className = "" }: { className?: string }) {
  const { data, error, isPending } = useSystemInfo();

  return (
    <Panel title="Platform build status" className={className}>
      {isPending ? (
        <p className="text-sm text-faint">Loading…</p>
      ) : error ? (
        <QueryError error={error} />
      ) : (
        <ol className="divide-y divide-line">
          {data.modules.map((module) => (
            <li key={module.key} className="flex items-start gap-4 py-3 first:pt-0 last:pb-0">
              <span className="num w-6 pt-0.5 text-xs text-faint">P{module.phase}</span>
              <div className="min-w-0 flex-1">
                <p className="text-sm text-text">{module.name}</p>
                <p className="mt-0.5 text-xs leading-5 text-muted">{module.summary}</p>
              </div>
              <ModuleStatusBadge status={module.status} />
            </li>
          ))}
        </ol>
      )}
    </Panel>
  );
}
