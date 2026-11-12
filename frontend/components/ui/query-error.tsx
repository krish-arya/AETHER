import { API_BASE_URL, ApiError } from "@/lib/api";

/** Honest failure state: says what failed and how to fix it; never substitutes data. */
export function QueryError({ error }: { error: unknown }) {
  const unreachable = error instanceof ApiError && error.unreachable;
  const message = error instanceof Error ? error.message : "Unknown error";
  return (
    <div role="alert" className="text-sm">
      <p className="text-down">{unreachable ? "Backend unreachable" : "Request failed"}</p>
      <p className="mt-1 text-muted">{message}</p>
      {unreachable ? (
        <p className="num mt-3 rounded border border-line bg-raised px-2 py-1.5 text-xs text-muted">
          Start the API ({API_BASE_URL}): uv run uvicorn app.main:app
        </p>
      ) : null}
    </div>
  );
}
