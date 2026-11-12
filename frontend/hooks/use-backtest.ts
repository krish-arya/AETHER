"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { fetchBacktestMetrics, fetchBacktestRuns, submitBacktest } from "@/services/backtest";

const RUNS_KEY = ["backtest", "runs"] as const;

/** Run list; polls while any job is still queued or running. */
export function useBacktestRuns() {
  return useQuery({
    queryKey: RUNS_KEY,
    queryFn: fetchBacktestRuns,
    retry: false,
    refetchInterval: (query) =>
      query.state.data?.some((run) => run.status === "queued" || run.status === "running")
        ? 2_000
        : false,
  });
}

export function useBacktestMetrics(runId: string | null) {
  return useQuery({
    queryKey: ["backtest", "metrics", runId],
    queryFn: () => fetchBacktestMetrics(runId as string),
    enabled: runId !== null,
    staleTime: Infinity, // results of a finished run never change
    retry: false,
  });
}

export function useSubmitBacktest() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: submitBacktest,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: RUNS_KEY }),
  });
}
