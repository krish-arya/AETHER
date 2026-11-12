// Thin fetch wrapper for the AETHER backend. No credentials are ever stored in the
// frontend: NEXT_PUBLIC_API_BASE_URL is a public address, inlined at build time.

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

export const API_PREFIX = "/api/v1";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** True when the backend could not be reached at all (as opposed to an HTTP error). */
  get unreachable(): boolean {
    return this.status === null;
  }
}

type GetOptions = {
  /** Non-2xx statuses whose JSON body is still a valid response (e.g. 503 readiness). */
  accept?: number[];
  params?: Record<string, string | number | undefined>;
};

export async function apiGet<T>(path: string, options: GetOptions = {}): Promise<T> {
  const url = new URL(`${API_BASE_URL}${API_PREFIX}${path}`);
  for (const [key, value] of Object.entries(options.params ?? {})) {
    if (value !== undefined) url.searchParams.set(key, String(value));
  }

  let response: Response;
  try {
    response = await fetch(url, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError(`Backend unreachable at ${API_BASE_URL}`, null);
  }
  if (!response.ok && !options.accept?.includes(response.status)) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // keep the status text
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

function errorDetail(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{ loc: [...], msg: "..." }, ...]
    return detail
      .map((item) => {
        const where = Array.isArray(item?.loc) ? item.loc.slice(1).join(".") : "";
        return where ? `${where}: ${item?.msg}` : String(item?.msg ?? "");
      })
      .join("; ");
  }
  return fallback;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${API_PREFIX}${path}`, {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(`Backend unreachable at ${API_BASE_URL}`, null);
  }
  if (!response.ok) {
    let parsed: unknown = null;
    try {
      parsed = await response.json();
    } catch {
      // keep the status text
    }
    throw new ApiError(errorDetail(parsed, response.statusText), response.status);
  }
  return (await response.json()) as T;
}
