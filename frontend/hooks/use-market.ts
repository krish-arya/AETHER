"use client";

import { useQuery } from "@tanstack/react-query";

import {
  fetchDatasets,
  fetchLatestIndicators,
  fetchStockDetail,
  fetchStocks,
} from "@/services/market";

const STALE = 60_000;

export function useDatasets() {
  return useQuery({
    queryKey: ["market", "datasets"],
    queryFn: fetchDatasets,
    staleTime: STALE,
    retry: false,
  });
}

export function useStocks() {
  return useQuery({
    queryKey: ["market", "stocks"],
    queryFn: fetchStocks,
    staleTime: STALE,
    retry: false,
  });
}

export function useStockDetail(symbol: string | null) {
  return useQuery({
    queryKey: ["market", "stock", symbol],
    queryFn: () => fetchStockDetail(symbol as string),
    enabled: symbol !== null,
    staleTime: STALE,
    retry: false,
  });
}

export function useLatestIndicators(symbol: string | null) {
  return useQuery({
    queryKey: ["market", "indicators", "latest", symbol],
    queryFn: () => fetchLatestIndicators(symbol as string),
    enabled: symbol !== null,
    staleTime: STALE,
    retry: false,
  });
}
