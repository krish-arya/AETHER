import { Badge, Dot } from "@/components/ui/badges";
import { Panel } from "@/components/ui/panel";

type PlannedModuleProps = {
  title: string;
  phase: number;
  phaseName: string;
  summary: string;
  willShow: string[];
  dependsOn: string[];
};

/**
 * Page body for a dashboard area whose engine has not been built yet.
 *
 * It states plainly that nothing here is implemented and lists what the page will
 * contain. It deliberately renders no charts, numbers or sample rows: placeholder
 * values in a finance UI are indistinguishable from real ones.
 */
export function PlannedModule({
  title,
  phase,
  phaseName,
  summary,
  willShow,
  dependsOn,
}: PlannedModuleProps) {
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Panel
        title={title}
        className="lg:col-span-2"
        aside={
          <Badge tone="muted">
            <Dot tone="muted" />
            Not built
          </Badge>
        }
      >
        <p className="max-w-2xl text-sm leading-6 text-text">{summary}</p>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
          This module is delivered in{" "}
          <span className="text-text">
            Phase {phase} — {phaseName}
          </span>
          . Until then this page shows no data, because there is none: no figures are
          simulated or stubbed.
        </p>

        <h3 className="mt-6 text-[11px] tracking-[0.14em] text-faint uppercase">
          Planned contents
        </h3>
        <ul className="mt-2 grid gap-x-6 gap-y-1.5 text-sm text-muted sm:grid-cols-2">
          {willShow.map((item) => (
            <li key={item} className="flex gap-2">
              <span aria-hidden className="text-faint">
                ·
              </span>
              {item}
            </li>
          ))}
        </ul>
      </Panel>

      <Panel title="Depends on">
        <ul className="space-y-2 text-sm text-muted">
          {dependsOn.map((item) => (
            <li key={item} className="flex gap-2">
              <span aria-hidden className="text-accent">
                →
              </span>
              {item}
            </li>
          ))}
        </ul>
      </Panel>
    </div>
  );
}
