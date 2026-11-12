"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { NAV_ITEMS, navItemFor } from "@/lib/nav";

export function Sidebar() {
  const pathname = usePathname();
  const active = navItemFor(pathname);

  return (
    <aside className="flex shrink-0 flex-col border-b border-line bg-panel md:w-60 md:border-r md:border-b-0">
      <div className="flex items-baseline gap-2 px-4 py-3.5 md:py-4">
        <span className="num text-base font-semibold tracking-[0.3em] text-accent">AETHER</span>
        <span className="text-[10px] tracking-wider text-faint uppercase">Research terminal</span>
      </div>

      <nav aria-label="Primary" className="flex gap-1 overflow-x-auto px-2 pb-2 md:flex-col md:pb-0">
        {NAV_ITEMS.map((item) => {
          const isActive = item.href === active?.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={isActive ? "page" : undefined}
              className={`group flex shrink-0 items-center gap-3 rounded px-2.5 py-2 text-[13px] transition-colors ${
                isActive ? "bg-raised text-text" : "text-muted hover:bg-raised/60 hover:text-text"
              }`}
            >
              <span
                className={`num w-8 text-[10px] tracking-wider ${isActive ? "text-accent" : "text-faint"}`}
              >
                {item.code}
              </span>
              <span className="flex-1 whitespace-nowrap">{item.label}</span>
              {item.live ? null : (
                <span
                  className="num hidden text-[9px] tracking-wider text-faint md:inline"
                  title={`Not built yet — roadmap phase ${item.phase}`}
                >
                  P{item.phase}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      <p className="mt-auto hidden px-4 py-4 text-[11px] leading-4 text-faint md:block">
        Simulation and research only. Nothing on these pages places a real order.
      </p>
    </aside>
  );
}
