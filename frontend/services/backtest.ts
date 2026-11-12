import { apiGet, apiPost } from "@/lib/api";
import type { BacktestMetrics, BacktestRequest, BacktestRun } from "@/lib/types";

export const fetchBacktestRuns = () => apiGet<BacktestRun[]>("/backtest/runs");

export const fetchBacktestMetrics = (runId: string) =>
  apiGet<BacktestMetrics>(`/backtest/runs/${encodeURIComponent(runId)}/metrics`);

export const submitBacktest = (request: BacktestRequest) =>
  apiPost<BacktestRun>("/backtest/run", request);
