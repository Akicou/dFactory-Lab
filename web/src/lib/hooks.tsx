import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Job } from "./api";

export function useApi<T>(path: string | null, intervalMs?: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  useEffect(() => {
    let alive = true;
    if (!path) return;
    (async () => {
      try { const d = await api<T>(path); if (alive) { setData(d); setError(null); } }
      catch (e) { if (alive) setError(String(e instanceof Error ? e.message : e)); }
    })();
    if (intervalMs) {
      const t = setInterval(() => setTick((x) => x + 1), intervalMs);
      return () => { alive = false; clearInterval(t); };
    }
    return () => { alive = false; };
  }, [path, intervalMs, tick]);
  return { data, error, reload };
}

export function useJobs() {
  return useApi<Job[]>("/api/jobs", 2500);
}

/** Submit a job and poll until terminal. Returns { run, job, running, error }. */
export function useJobRunner() {
  const [job, setJob] = useState<Job | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const run = useCallback(async (path: string, body: unknown) => {
    setRunning(true); setError(null); setJob(null);
    try {
      const res = await api<{ job_id: string }>(path, { method: "POST", body: JSON.stringify(body) });
      const id = res.job_id;
      const poll = async () => {
        try {
          const j = await api<Job>(`/api/jobs/${id}`);
          setJob(j);
          if (j.state === "running" || j.state === "queued") {
            timer.current = window.setTimeout(poll, 800);
          } else {
            setRunning(false);
            if (j.state === "error") setError(j.error ?? "job failed");
          }
        } catch (e) { setRunning(false); setError(String(e instanceof Error ? e.message : e)); }
      };
      poll();
    } catch (e) {
      setRunning(false);
      setError(String(e instanceof Error ? e.message : e));
    }
  }, []);

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  return { run, job, running, error };
}

export function fmtBytes(n: number): string {
  if (!n) return "0 B";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"];
  const i = Math.floor(Math.log(n) / Math.log(1024));
  return `${(n / Math.pow(1024, i)).toFixed(i ? 1 : 0)} ${u[i]}`;
}

export function JobPill({ state }: { state: string }) {
  const color = state === "done" ? "text-ok" : state === "error" ? "text-danger"
    : state === "running" ? "text-accent-2" : "text-faint";
  return <span className={`badge mono ${color}`}>{state}</span>;
}
