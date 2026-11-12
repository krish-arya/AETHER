"use client";

import { Suspense } from "react";
import { usePathname } from "next/navigation";

import { Badge, Dot } from "@/components/ui/badges";
import { useReadiness } from "@/hooks/use-system";
import { navItemFor } from "@/lib/nav";

function BackendStatus() {
  const { data, error, isPending } = useReadiness();

  if (isPending) return <Badge tone="muted">Backend · checking</Badge>;
  if (error || !data) {
    return (
      <Badge tone="down">
        <Dot tone="down" />
        Backend · offline
      </Badge>
    );
  }
  if (data.status === "ok") {
    return (
      <Badge tone="ok">
        <Dot tone="ok" />
        Backend · ready
      </Badge>
    );
  }
  const down = data.components.filter((c) => c.status === "down").map((c) => c.name);
  return (
    <span title={`Unavailable: ${down.join(", ")}`}>
      <Badge tone="warn">
        <Dot tone="warn" />
        Backend · degraded ({down.join(", ")})
      </Badge>
    </span>
  );
}

export function Topbar() {
  const item = navItemFor(usePathname());

  return (
    <header className="flex items-center justify-between gap-4 border-b border-line px-4 py-3 md:px-6">
      <div className="flex min-w-0 items-baseline gap-3">
        <h1 className="truncate text-[15px] font-medium text-text">{item?.label ?? "AETHER"}</h1>
        {item && !item.live ? (
          <span className="num text-[10px] tracking-wider text-faint uppercase">
            Phase {item.phase} · not built
          </span>
        ) : null}
      </div>
      <Suspense fallback={<Badge tone="muted">Backend · checking</Badge>}>
        <BackendStatus />
      </Suspense>
    </header>
  );
}
