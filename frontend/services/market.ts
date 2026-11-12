import { apiGet } from "@/lib/api";
import type { DatasetVersion, Indicators, Security, SecurityDetail } from "@/lib/types";

export const fetchDatasets = () => apiGet<DatasetVersion[]>("/market/datasets");

export const fetchStocks = () => apiGet<Security[]>("/market/stocks");

export const fetchStockDetail = (symbol: string) =>
  apiGet<SecurityDetail>(`/market/stocks/${encodeURIComponent(symbol)}`);

export const fetchLatestIndicators = (symbol: string) =>
  apiGet<Indicators>(`/market/indicators/${encodeURIComponent(symbol)}`, {
    params: { limit: 1 },
  });
