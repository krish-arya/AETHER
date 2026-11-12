// Mirrors the backend response schemas (backend/app/schemas). Keep in sync.

export type ComponentStatus = {
  name: string;
  status: "ok" | "down";
  latency_ms: number | null;
  detail: string | null;
};

export type Readiness = {
  status: "ok" | "degraded";
  components: ComponentStatus[];
  time: string;
};

export type ModuleStatus = "available" | "in_progress" | "planned";

export type ModuleInfo = {
  key: string;
  name: string;
  phase: number;
  status: ModuleStatus;
  summary: string;
};

export type SystemInfo = {
  name: string;
  version: string;
  environment: string;
  database_backend: string;
  disclaimers: string[];
  modules: ModuleInfo[];
};

export type DatasetVersion = {
  dataset_id: string;
  name: string;
  provider: string;
  is_synthetic: boolean;
  license_note: string;
  content_hash: string;
  config_hash: string;
  feature_set_version: string;
  start_date: string;
  end_date: string;
  as_of: string;
  symbol_count: number;
  bar_count: number;
  quality_summary: {
    rows_in: number;
    rows_out: number;
    rows_quarantined: number;
    quarantine_fraction: number;
    by_severity: Record<string, number>;
    by_check: Record<string, number>;
  };
  created_at: string;
  loaded_at: string;
};

export type Security = {
  symbol: string;
  name: string;
  sector: string | null;
  industry: string | null;
  isin: string | null;
  listing_date: string | null;
  delisting_date: string | null;
  is_synthetic: boolean;
  source: string;
  dataset_id: string;
};

export type Bar = {
  trade_date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  adjusted_close: number;
  volume: number;
  turnover: number | null;
  delivery_volume: number | null;
  market_cap: number | null;
};

export type SecurityDetail = Security & {
  first_trade_date: string | null;
  last_trade_date: string | null;
  bar_count: number;
  latest_bar: Bar | null;
};

export type Indicators = {
  symbol: string;
  is_synthetic: boolean;
  feature_set_version: string;
  dataset_id: string;
  feature_names: string[];
  count: number;
  rows: { trade_date: string; features: Record<string, number | null> }[];
};

export type BacktestStatus = "queued" | "running" | "succeeded" | "failed";

export type BacktestRun = {
  run_id: string;
  name: string;
  status: BacktestStatus;
  dataset_id: string;
  is_synthetic: boolean;
  result_id: string | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

export type MetricSet = Record<string, number | null>;

export type BacktestMetrics = {
  run_id: string;
  status: string;
  benchmark: string;
  is_synthetic: boolean;
  caveats: string[];
  full_period: Record<string, MetricSet>;
  walk_forward: Record<string, MetricSet>;
};

export type BacktestRequest = {
  name?: string;
  initial_capital?: number;
  rebalance?: "daily" | "weekly" | "monthly" | "quarterly";
};
