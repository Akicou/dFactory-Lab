// Thin fetch client for the dFactory-Lab backend. Every response uses the
// { ok, data } | { ok: false, error } envelope.
const BASE = import.meta.env.VITE_API_BASE ?? "";

export type ApiResult<T> = { ok: true; data: T } | { ok: false; error: string };

export async function api<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  const body = await res.json().catch(() => ({ ok: false, error: res.statusText }));
  if (!res.ok || body?.ok === false) {
    throw new Error(body?.error ?? `HTTP ${res.status}`);
  }
  return body.data as T;
}

export const apiRaw = async (path: string, init?: RequestInit) =>
  fetch(`${BASE}${path}`, init);

// ── typed shapes ────────────────────────────────────────────────────────────
export interface Health {
  status: string; version: string; phase: number; bind: string;
  uptime_s: number; active_jobs: number; gpu: { backend: string; device_count: number } | null;
}
export interface SystemInfo {
  python: string; platform: string; cpus: number;
  hardware: { backend: string; device_count: number };
  data_dir: string; engine_present: boolean;
}
export interface Job {
  id: string; kind: string; state: string; progress: number; message: string;
  created_at: string; started_at: string | null; ended_at: string | null;
  result: unknown; error: string | null;
}
export interface ModelMeta {
  id: string; repo_id: string; name: string; size: string;
  config_path: string; meta: Record<string, number | string>;
}
export interface LocalModel {
  id: string; path: string; format: string; size_bytes: number;
  has_modeling: boolean; num_experts?: number;
}
export interface DatasetRow { id: string; name: string; source: string; path: string; rows: number; created_at: string; }
