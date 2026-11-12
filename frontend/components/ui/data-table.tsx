import type { ReactNode } from "react";

import { ApiError } from "@/lib/api";

import { QueryError } from "./query-error";

export type Column<T> = {
  label: string;
  align?: "left" | "right";
  render: (row: T) => ReactNode;
};

type DataTableProps<T> = {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T, index: number) => string;
  empty?: string;
};

/** Dense read-only table. Numeric columns are right-aligned and tabular. */
export function DataTable<T>({ columns, rows, rowKey, empty = "Nothing to show." }: DataTableProps<T>) {
  if (rows.length === 0) return <p className="text-sm text-muted">{empty}</p>;
  return (
    <div className="-mx-4 overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line text-[11px] tracking-wide text-faint uppercase">
            {columns.map((column) => (
              <th
                key={column.label}
                className={`px-4 py-2 font-normal whitespace-nowrap ${column.align === "right" ? "text-right" : "text-left"}`}
              >
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={rowKey(row, index)} className="border-b border-line/60 last:border-0">
              {columns.map((column) => (
                <td
                  key={column.label}
                  className={`px-4 py-2 align-top ${column.align === "right" ? "num text-right text-text" : "text-muted"}`}
                >
                  {column.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Loading / error / "not there yet" states for a query. A 404 from the API means
 * "this does not exist yet" (no model trained, no paper account) and is shown as a
 * plain message rather than as a failure.
 */
export function QueryState({
  isPending,
  error,
  children,
}: {
  isPending: boolean;
  error: unknown;
  children: ReactNode;
}) {
  if (isPending) return <p className="text-sm text-faint">Loading…</p>;
  if (error instanceof ApiError && error.status === 404) {
    return <p className="text-sm leading-6 text-muted">{error.message}</p>;
  }
  if (error) return <QueryError error={error} />;
  return <>{children}</>;
}

export function Caveats({ items }: { items: string[] }) {
  return (
    <ul className="space-y-1.5">
      {items.map((item) => (
        <li
          key={item}
          className="rounded border border-warn/30 bg-warn/5 px-3 py-2 text-xs leading-5 text-warn"
        >
          {item}
        </li>
      ))}
    </ul>
  );
}

export const buttonClass =
  "cursor-pointer rounded border border-accent/60 px-3 py-2 text-sm text-accent transition-colors hover:bg-accent/10 disabled:cursor-wait disabled:opacity-50";
export const fieldClass =
  "num mt-1 w-full rounded border border-line bg-raised px-2 py-1.5 text-sm text-text";
export const labelClass = "block text-[11px] tracking-wide text-faint uppercase";
export const headingClass = "text-[11px] tracking-[0.14em] text-faint uppercase";
