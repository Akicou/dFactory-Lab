import { useEffect, useState } from "react";
import { Activity, Cpu, Layers, CheckCircle2, Download, Merge, FlaskConical, Split, MessageSquare } from "lucide-react";
import { api, type Health, type SystemInfo, type Job } from "../lib/api";
import { Page } from "../components/Page";

const STEPS = [
  { icon: Download, label: "Download" },
  { icon: Merge, label: "Merge", note: "auto" },
  { icon: FlaskConical, label: "Train" },
  { icon: Split, label: "Split" },
  { icon: MessageSquare, label: "Chat" },
];

export default function Dashboard() {
  const [health, setHealth] = useState<Health | null>(null);
  const [sys, setSys] = useState<SystemInfo | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try { const h = await api<Health>("/api/health"); if (alive) setHealth(h); } catch { /* server starting */ }
    };
    poll();
    api<SystemInfo>("/api/system").then((s) => alive && setSys(s)).catch(() => {});
    api<Job[]>("/api/jobs").then((j) => alive && setJobs(j)).catch(() => {});
    const t = setInterval(poll, 4000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  return (
    <Page title="Dashboard">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 anim-stagger">
        <Stat icon={Activity} label="Server" value={health?.status ?? "—"} ok={health?.status === "ok"} />
        <Stat icon={Cpu} label="Hardware" value={health?.gpu ? `${health.gpu.backend} · ${health.gpu.device_count}` : "—"} />
        <Stat icon={Layers} label="Active jobs" value={String(health?.active_jobs ?? 0)} />
        <Stat icon={CheckCircle2} label="Engine" value={sys?.engine_present ? "ready" : "—"} ok={sys?.engine_present} />
      </div>

      <div>
        <div className="label mb-3">Pipeline</div>
        <div className="flex flex-wrap items-center gap-2">
          {STEPS.map((s, i) => (
            <div key={s.label} className="flex items-center gap-2">
              <div className="card card-hover px-3 py-2 flex items-center gap-2">
                <s.icon size={15} strokeWidth={1.9} className="text-muted" />
                <span className="text-sm">{s.label}</span>
                {s.note && <span className="badge">{s.note}</span>}
              </div>
              {i < STEPS.length - 1 && <span className="text-faint">→</span>}
            </div>
          ))}
        </div>
      </div>

      <div className="card p-5">
        <div className="label mb-3">Recent jobs</div>
        {jobs.length === 0 ? (
          <div className="text-sm text-faint py-4 text-center">No jobs yet.</div>
        ) : (
          <div className="divide-y divide-edge">
            {jobs.slice(0, 6).map((j) => (
              <div key={j.id} className="flex items-center gap-3 py-2.5">
                <span className="badge w-20 justify-center">{j.kind}</span>
                <span className={`badge w-24 justify-center ${j.state === "done" ? "!text-ok" : j.state === "error" ? "!text-danger" : j.state === "running" ? "!text-warn pulse" : ""}`}>{j.state}</span>
                <span className="text-sm text-muted flex-1 truncate">{j.message}</span>
                <span className="mono text-xs text-faint">{Math.round(j.progress * 100)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </Page>
  );
}

function Stat({ icon: Icon, label, value, ok }: { icon: typeof Activity; label: string; value: string; ok?: boolean }) {
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between">
        <span className="label">{label}</span>
        <Icon size={15} strokeWidth={1.9} className="text-faint" />
      </div>
      <div className={`mt-2 text-lg font-semibold capitalize ${ok === false ? "text-danger" : ""}`}>{value}</div>
    </div>
  );
}
