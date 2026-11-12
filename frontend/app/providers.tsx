"use client";

import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

let browserQueryClient: QueryClient | undefined;

function getQueryClient() {
  // Isolate server renders; keep one cache for the lifetime of the browser tab.
  if (typeof window === "undefined") return new QueryClient();
  browserQueryClient ??= new QueryClient({
    defaultOptions: { queries: { refetchOnWindowFocus: false } },
  });
  return browserQueryClient;
}

export function Providers({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={getQueryClient()}>{children}</QueryClientProvider>;
}
