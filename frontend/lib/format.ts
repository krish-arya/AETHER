// Display formatting. A missing value is always shown as an em dash -- never as 0.

export const MISSING = "—";

const integer = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
const price = new Intl.NumberFormat("en-IN", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function formatInteger(value: number | null | undefined): string {
  return value == null ? MISSING : integer.format(value);
}

export function formatPrice(value: number | null | undefined): string {
  return value == null ? MISSING : price.format(value);
}

export function formatNumber(value: number | null | undefined, digits = 2): string {
  return value == null ? MISSING : value.toFixed(digits);
}

/** Format a fraction (0.0123) as a signed percentage ("+1.23%"). */
export function formatPercent(value: number | null | undefined, digits = 2): string {
  if (value == null) return MISSING;
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(digits)}%`;
}

/** Indian-style large amounts: crore (1e7) and lakh crore. */
export function formatCrore(value: number | null | undefined): string {
  if (value == null) return MISSING;
  return `₹${integer.format(Math.round(value / 1e7))} cr`;
}
