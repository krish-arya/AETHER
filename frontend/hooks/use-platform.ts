"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { apiGet, apiPost } from "@/lib/api";
import type {
  AllocationRecord,
  ExperimentReport,
  GanEvaluation,
  JobRow,
  ModelCard,
  NewsList,
  PaperOrders,
  PaperPerformance,
  PaperPositions,
  PaperSummary,
  RegimeResponse,
  RiskMetrics,
  ScenarioSet,
} from "@/lib/platform-types";
import type { DatasetVersion } from "@/lib/types";

const options = { retry: false, staleTime: 30_000 } as const;

function useGet<T>(key: string[], path: string, enabled = true, params?: Record<string, string | number>) {
  return useQuery({ queryKey: key, queryFn: () => apiGet<T>(path, { params }), enabled, ...options });
}

export const useLatestDataset = () =>
  useQuery({
    queryKey: ["market", "datasets", "latest"],
    queryFn: async () => (await apiGet<DatasetVersion[]>("/market/datasets"))[0] ?? null,
    ...options,
  });

export const useRegime = (limit = 250) =>
  useGet<RegimeResponse>(["market", "regime", String(limit)], "/market/regime", true, { limit });

export const useNews = () => useGet<NewsList>(["news"], "/news", true, { limit: 50 });

export const useModels = (kind: "rl" | "gan") => useGet<ModelCard[]>(["models", kind], `/${kind}/models`);

export const useGanEvaluation = (modelId: string | null) =>
  useGet<GanEvaluation>(["gan", "evaluation", modelId ?? ""], `/gan/evaluation/${modelId}`, modelId !== null);

export const useExperiments = () => useGet<ExperimentReport[]>(["experiments"], "/experiments");

export const usePaperPositions = () => useGet<PaperPositions>(["paper", "positions"], "/paper/positions");
export const usePaperPerformance = (enabled: boolean) =>
  useGet<PaperPerformance>(["paper", "performance"], "/paper/performance", enabled);
export const usePaperOrders = (enabled: boolean) =>
  useGet<PaperOrders>(["paper", "orders"], "/paper/orders", enabled);
export const useRiskMetrics = (enabled: boolean) =>
  useGet<RiskMetrics>(["risk", "metrics"], "/risk/metrics", enabled);

export function useJobs() {
  return useQuery({
    queryKey: ["jobs"],
    queryFn: () => apiGet<JobRow[]>("/jobs", { params: { limit: 20 } }),
    retry: false,
    refetchInterval: (query) =>
      query.state.data?.some((job) => job.status === "queued" || job.status === "running")
        ? 3_000
        : false,
  });
}

function usePost<TBody, TResult>(path: string, invalidate: string[][]) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: TBody) => apiPost<TResult>(path, body),
    onSuccess: () => {
      for (const queryKey of invalidate) client.invalidateQueries({ queryKey });
    },
  });
}

export const useAllocate = () =>
  usePost<{ model_id?: string }, AllocationRecord>("/rl/allocate", [["portfolio"]]);

export const useGenerateScenarios = () =>
  usePost<{ regime: string; scenarios: number }, ScenarioSet>("/gan/generate", [["gan", "scenarios"]]);

export const useSubmitJob = (path: string) => usePost<Record<string, unknown>, JobRow>(path, [["jobs"]]);

export type InitializeBody = {
  name: string;
  initial_capital: number;
  strategy: { kind: string; ref?: string };
  rebalance_every: number;
};

export const useInitializePaper = () =>
  usePost<InitializeBody, PaperSummary>("/paper/initialize", [["paper"], ["risk"], ["portfolio"]]);

export const usePlaceOrder = () =>
  usePost<{ symbol: string; target_weight: number }, { status: string; detail: string | null }>(
    "/paper/orders",
    [["paper"], ["risk"]],
  );

export type AlertItem = { severity: "critical" | "warning" | "info"; code: string; message: string };

export const useAlerts = () =>
  useGet<{ checked_at: string; alerts: AlertItem[] }>(["system", "alerts"], "/system/alerts");
