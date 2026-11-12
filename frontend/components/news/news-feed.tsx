"use client";

import { useState } from "react";

import { Badge, type Tone } from "@/components/ui/badges";
import { QueryState, fieldClass, labelClass } from "@/components/ui/data-table";
import { Panel } from "@/components/ui/panel";
import { useNews } from "@/hooks/use-platform";
import { formatNumber } from "@/lib/format";
import type { NewsItem } from "@/lib/platform-types";

const SENTIMENT_TONE: Record<string, Tone> = {
  positive: "ok",
  negative: "down",
  mixed: "warn",
  uncertain: "warn",
  neutral: "muted",
};

function Article({ item }: { item: NewsItem }) {
  return (
    <article className="py-4 first:pt-0 last:pb-0">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="text-sm text-text">
          {item.url ? (
            <a href={item.url} target="_blank" rel="noreferrer" className="hover:text-accent">
              {item.headline}
            </a>
          ) : (
            item.headline
          )}
        </h3>
      </div>
      <p className="num mt-1 text-[11px] text-faint">
        {item.source} · {item.published_at.slice(0, 16).replace("T", " ")} UTC ·{" "}
        {item.symbols.join(", ") || "no symbol"} · {item.has_body ? "full text" : "headline only"}
      </p>
      {item.events.length === 0 ? (
        <p className="mt-2 text-xs text-muted">No validated event was extracted from this article.</p>
      ) : (
        <ul className="mt-2 space-y-2">
          {item.events.map((event) => (
            <li key={event.event_id} className="rounded border border-line bg-raised px-3 py-2">
              <div className="flex flex-wrap items-center gap-2">
                <span className="num text-xs text-text">{event.symbol}</span>
                <Badge tone="muted">{event.event_type.replaceAll("_", " ")}</Badge>
                <Badge tone={SENTIMENT_TONE[event.sentiment] ?? "muted"}>
                  {event.sentiment} {formatNumber(event.sentiment_score, 2)}
                </Badge>
                <span className="num text-[11px] text-faint">
                  impact {event.impact_direction.replaceAll("_", " ")} ·{" "}
                  {event.impact_horizon.replaceAll("_", " ")} · severity{" "}
                  {formatNumber(event.severity, 2)} · confidence {formatNumber(event.confidence, 2)}
                </span>
              </div>
              {event.key_facts.length ? (
                <p className="mt-1.5 text-xs leading-5 text-muted">{event.key_facts.join(" · ")}</p>
              ) : null}
              {event.flags.length ? (
                <p className="mt-1 text-[11px] text-warn">
                  Flagged: {event.flags.map((flag) => flag.flag.replaceAll("_", " ")).join(", ")}
                </p>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}

export function NewsFeed() {
  const { data, error, isPending } = useNews();
  const [symbol, setSymbol] = useState("");
  const [eventType, setEventType] = useState("");

  const items = data?.items ?? [];
  const symbols = [...new Set(items.flatMap((item) => item.symbols))].sort();
  const eventTypes = [...new Set(items.flatMap((item) => item.events.map((e) => e.event_type)))].sort();
  const shown = items.filter(
    (item) =>
      (!symbol || item.symbols.includes(symbol)) &&
      (!eventType || item.events.some((event) => event.event_type === eventType)),
  );
  const model = data?.models[0];

  return (
    <Panel
      title="Financial events"
      aside={
        model ? (
          <span className="num">
            {model.provider} · {model.model} · {model.prompt_version}
          </span>
        ) : null
      }
    >
      <QueryState isPending={isPending} error={error}>
        {items.length === 0 ? (
          <div className="text-sm text-muted">
            <p>No analysed news has been loaded into the database yet.</p>
            <pre className="num mt-3 overflow-x-auto rounded border border-line bg-raised p-3 text-xs leading-5">
              {"uv run python scripts/analyze_news.py --max-articles 20\nuv run python scripts/load_news.py --latest"}
            </pre>
          </div>
        ) : (
          <>
            <div className="mb-4 grid max-w-md grid-cols-2 gap-3">
              <label className={labelClass}>
                Company
                <select className={fieldClass} value={symbol} onChange={(e) => setSymbol(e.target.value)}>
                  <option value="">All</option>
                  {symbols.map((s) => (
                    <option key={s}>{s}</option>
                  ))}
                </select>
              </label>
              <label className={labelClass}>
                Event type
                <select
                  className={fieldClass}
                  value={eventType}
                  onChange={(e) => setEventType(e.target.value)}
                >
                  <option value="">All</option>
                  {eventTypes.map((t) => (
                    <option key={t}>{t}</option>
                  ))}
                </select>
              </label>
            </div>
            <div className="divide-y divide-line">
              {shown.map((item) => (
                <Article key={item.news_id} item={item} />
              ))}
            </div>
            <p className="mt-4 text-xs leading-5 text-faint">
              Events are extracted by a language model and kept only if they pass schema, entity,
              timestamp, source and grounding checks. They summarise what the article says; they are
              not verified facts and not advice.
            </p>
          </>
        )}
      </QueryState>
    </Panel>
  );
}
