import { apiGet } from "@/lib/api";
import type { Readiness, SystemInfo } from "@/lib/types";

// 503 carries a normal readiness body ("degraded"), so it is not an error here.
export const fetchReadiness = () => apiGet<Readiness>("/health/ready", { accept: [503] });

export const fetchSystemInfo = () => apiGet<SystemInfo>("/system/info");
