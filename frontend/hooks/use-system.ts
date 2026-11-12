"use client";

import { useQuery } from "@tanstack/react-query";

import { fetchReadiness, fetchSystemInfo } from "@/services/system";

export function useReadiness() {
  return useQuery({
    queryKey: ["system", "readiness"],
    queryFn: fetchReadiness,
    refetchInterval: 15_000,
    retry: false,
  });
}

export function useSystemInfo() {
  return useQuery({
    queryKey: ["system", "info"],
    queryFn: fetchSystemInfo,
    staleTime: 5 * 60_000,
    retry: false,
  });
}
