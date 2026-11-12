"use client";

import { Badge, Dot } from "@/components/ui/badges";
import { Panel, Stat } from "@/components/ui/panel";
import { QueryError } from "@/components/ui/query-error";
import { useReadiness, useSystemInfo } from "@/hooks/use-system";
import { formatNumber } from "@/lib/format";

const COMPONENT_LABEL: Record<string, string> = {
  database: "Database",
  redis: "Redis (cache / job queue)",
};

export function SystemStatusPanel() {
  const readiness = useReadiness();
  const info = useSystemInfo();

  return (
    <Panel
      title="System status"
      aside={info.data ? <span className="num">v{info.data.version}</span> : null}
    >
      {readiness.isPending ? (
        <p className="text-sm text-faint">Checking backend…</p>
      ) : readiness.error ? (
        <QueryError error={readiness.error} />
      ) : (
        <ul className="divide-y divide-line">
          <li className="flex items-center justify-between py-2 first:pt-0">
            <span className="text-sm text-text">API</span>
            <Badge tone="ok">
              <Dot tone="ok" />
              Up
            </Badge>
          </li>
          {readiness.data.components.map((component) => (
            <li key={component.name} className="flex items-center justify-between gap-3 py-2">
              <span className="text-sm text-text">
                {COMPONENT_LABEL[component.name] ?? component.name}
              </span>
              <span className="flex items-center gap-3">
                {component.latency_ms != null ? (
                  <span className="num text-xs text-faint">
                    {formatNumber(component.latency_ms, 1)} ms
                  </span>
                ) : null}
                {component.status === "ok" ? (
                  <Badge tone="ok">
                    <Dot tone="ok" />
                    Up
                  </Badge>
                ) : (
                  <span title={component.detail ?? undefined}>
                    <Badge tone="down">
                      <Dot tone="down" />
                      Down
                    </Badge>
                  </span>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}

      {info.data ? (
        <dl className="mt-4 grid grid-cols-2 gap-3 border-t border-line pt-4">
          <Stat label="Environment">{info.data.environment}</Stat>
          <Stat label="Database engine">{info.data.database_backend}</Stat>
        </dl>
      ) : null}
    </Panel>
  );
}
