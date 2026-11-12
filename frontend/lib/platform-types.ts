// Response shapes of the model-backed endpoints (regime, news, GAN, RL, risk, paper).
// Mirrors backend/app/api/v1/{market,news,models_api,portfolio}.py. Keep in sync.

export type Num = number | null;

export type RegimePoint = {
  timestamp: string;
  regime: string | null;
  confidence: Num;
  volatility_level: Num;
  trend_strength: Num;
  realized_vol: Num;
  trend_return: Num;
  drawdown: Num;
};

export type RegimeResponse = {
  dataset_id: string;
  is_synthetic: boolean;
  method: string;
  current: RegimePoint | null;
  history: RegimePoint[];
};

export type LlmInfo = { provider: string; model: string; prompt_version: string };

export type NewsEvent = {
  event_id: string;
  symbol: string;
  company: string;
  event_type: string;
  sentiment: string;
  sentiment_score: number;
  impact_direction: string;
  impact_horizon: string;
  severity: number;
  confidence: number;
  key_facts: string[];
  risks: string[];
  source_ids: string[];
  flags: { flag: string; detail: string }[];
};

export type NewsItem = {
  news_id: string;
  headline: string;
  has_body: boolean;
  source: string;
  url: string | null;
  published_at: string;
  symbols: string[];
  llm: LlmInfo;
  events: NewsEvent[];
};

export type NewsList = { count: number; models: LlmInfo[]; items: NewsItem[] };

export type Gate = { value: number; threshold: number; rule: string; passed: boolean };

export type ModelCard = {
  model_id: string;
  kind: string;
  status: "candidate" | "validated" | "rejected";
  trained_at: string;
  dataset_id: string | null;
  trained_on_synthetic_data: boolean | null;
  seed: number | null;
  notes: string | null;
  lineage: { experiment?: string | null; generator_id?: string | null; fold?: number };
  training_window: Record<string, string[] | string> | null;
  metrics: Record<string, unknown> & {
    validation_sharpe?: Num;
    passes?: boolean;
    gates?: Record<string, Gate>;
  };
};

type Comparison = {
  return_distribution: { ks: number; wasserstein: number; real_std: number; other_std: number };
  volatility_distribution: { ks: number };
  autocorrelation: { returns_error: number; abs_returns_error: number };
  cross_correlation: { mean_abs_error: number; real_mean: number; other_mean: number };
  drawdown_distribution: { ks: number; real_mean: number; other_mean: number };
  tail_risk: {
    real: { q05: number; cvar05: number; excess_kurtosis: number };
    other: { q05: number; cvar05: number; excess_kurtosis: number };
  };
};

export type GanEvaluation = {
  passes: boolean;
  gates: Record<string, Gate>;
  windows: { train: number; heldout_real: number; synthetic: number };
  synthetic_vs_heldout_real: Comparison;
  real_baseline_train_vs_heldout: Comparison;
  discriminative: { score: number } & Record<string, { accuracy: number; score: number } | number>;
  memorization: { ratio: number; copied_fraction: number; closer_than_5pct_of_real: number };
};

export type ScenarioSet = {
  scenario_set_id: string;
  created_at: string;
  label: string;
  generator_id: string;
  generator_status: string;
  regime: string;
  purpose: string;
  scenarios: number;
  horizon_sessions: number;
  equal_weight_path: { p05: number[]; p50: number[]; p95: number[] };
  equal_weight_total_return: { mean: number; p05: number; worst: number };
  assets: { symbol: string; mean_return: number; p05_return: number; annualised_volatility: number }[];
};

export type RiskDecision = {
  rejected: boolean;
  reason: string | null;
  applied: string[];
  raw_violations: { limit: string; value: number; threshold: number; subject: string | null }[];
  turnover: number;
  cash: Num;
};

export type AllocationRecord = {
  allocation_id: string;
  created_at: string;
  as_of: string;
  executable_from: string | null;
  model_id: string;
  dataset_id: string;
  is_synthetic: boolean;
  summary: {
    regime: { regime: string | null; volatility_level: Num; trend_strength: Num } | null;
    invested: number;
    cash: number;
    positions: number;
    turnover: number;
    narrative: string[];
  };
  allocation: Record<string, number>;
  previous_allocation: Record<string, number>;
  evidence: {
    positions: {
      symbol: string;
      weight: number;
      previous_weight: number;
      change: number;
      policy_weight_before_risk: number;
      sector: string | null;
      most_influential_features: { feature: string; value: Num; weight_change_if_average: number }[];
    }[];
    risk_engine: RiskDecision;
    risk_limits: Record<string, Num>;
    policy_action_spread: Num;
    attention_by_source: Record<string, number> | null;
  };
  sensitivity: { by_information_source: Record<string, number>; definition: string; note: string };
  model: {
    trained_at: string;
    status: string;
    experiment: string | null;
    validation_sharpe: Num;
    notes: string | null;
    training_window: Record<string, string[]>;
  };
};

export type ExperimentEntry = {
  label: string;
  skipped?: string;
  kind?: string;
  seeds?: number[];
  mean?: Record<string, Num>;
  std?: Record<string, Num>;
  min?: Record<string, Num>;
  max?: Record<string, Num>;
  risk_engine?: { decisions: number; modified: number; rejected: number } | null;
  sharpe_difference_vs_E0?: {
    estimate: number;
    ci_low: number;
    ci_high: number;
    p_value: number;
    significant: boolean;
  } | null;
};

export type ExperimentReport = {
  run_id: string;
  created_at: string;
  dataset: { dataset_id: string; is_synthetic: boolean };
  folds: { index: number; test_start: string; test_end: string }[];
  seeds: number[];
  caveats: string[];
  duration_seconds: number;
  experiments: Record<string, ExperimentEntry>;
  generator: { fold: number; model_id: string; passes: boolean; gates: Record<string, Gate> }[];
};

export type JobRow = {
  job_id: string;
  kind: string;
  status: "queued" | "running" | "succeeded" | "failed";
  params: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  finished_at: string | null;
};

export type PaperSummary = {
  account_id: string;
  name: string;
  strategy: { kind: string; ref: string | null };
  dataset_id: string;
  is_synthetic: boolean;
  start_date: string;
  as_of: string;
  initial_capital: number;
  value: number;
  cash: number;
  invested: number;
  daily_return: number;
  cumulative_return: number;
  drawdown: number;
  positions: number;
  note: string;
};

export type PaperHolding = {
  symbol: string;
  weight: number;
  value: number;
  sector: string | null;
  last_close: Num;
};

export type PaperPositions = PaperSummary & { holdings: PaperHolding[] };

export type PaperPerformance = {
  as_of: string;
  metrics: Record<string, Num>;
  slippage_cost: number;
  equity: { dates: string[]; values: number[] };
  daily_pnl: { date: string; pnl: number; return: number }[];
};

export type PaperOrders = {
  manual_orders: {
    order_id: string;
    decision_date: string;
    symbol: string;
    target_weight: number;
    status: string;
    detail: string | null;
  }[];
  executed_trades: {
    trade_date: string;
    symbol: string;
    side: string;
    value: number;
    price: number;
    total_cost: number;
    realized_pnl: Num;
  }[];
  rejected_or_limited: { trade_date: string; symbol: string | null; event: string; detail: string }[];
};

export type RiskMetrics = {
  source: string;
  as_of: string;
  window_sessions: number;
  var: Num;
  cvar: Num;
  confidence_level: number;
  volatility: Num;
  max_drawdown: Num;
  current_drawdown: number;
  beta?: Num;
  cash: number;
  concentration?: { largest_sector: string | null; largest_sector_weight: number; herfindahl: number };
  stress?: { worst: Num };
  limit_violations: { limit: string; value: number; threshold: number; subject: string | null }[];
  note: string;
};
