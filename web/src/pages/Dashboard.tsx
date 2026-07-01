import { useEffect, useState } from "react";
import { api, type Health, type SystemInfo } from "@/lib/api";

const PIPELINE = [
  { n: 1, label: "Download", to: "/models" },
  { n: 2, label: "MoE merge", to: "/models" },
  { n: 3, label: "Build dataset", to: "/datasets" },
  { n: 4, label: "Train (BD-SFT)", to: "/training" },
  { n: 5, label: "MoE split", to: "/export" },
  { n: 6, label: "Copy modeling file", to: "/export" },
  { n: 7, label: "Package", to: "/export" },
  { n: 8, label: "Chat", to: "/chat" },
];

export default function Dashboard() {
  const [health, setHealth] = useState<Health | null>(null);
  const [sys, setSys] = useState<SystemInfo | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const h = await api<Health>("/api/health");
        if (alive) { setHealth(h); setErr(null); }
      } catch (e) { if (alive) setErr(String(e instanceof Error ? e.message : e)); }
    };
    poll();
    api<SystemInfo>("/api/system").then(setSys).catch(() => {});
    const t = setInterval(poll, 4000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  return (
    <div className="space-y-9">
      <header>
        <div className="eyebrow">Phase 7 · Editorial Paper</div>
        <h1 className="text-3xl mt-1">Fine-tune diffusion LLMs, without the CLI.</h1>
        <p className="text-muted mt-2 max-w-2xl leading-relaxed">
          dFactory-Lab wraps the dFactory engine — download, MoE expert merge, dataset prep,
          block-diffusion SFT, split, packaging and chat — behind one lab console.
        </p>
      </header>

      {/* status strip */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Server" value={health?.status ?? "—"} ok={health?.status === "ok"} />
        <Stat label="Hardware" value={health?.gpu ? `${health.gpu.backend} ×${health.gpu.device_count}` : "—"} />
        <Stat label="Active jobs" value={String(health?.active_jobs ?? 0)} />
        <Stat label="Engine vendored" value={sys?.engine_present ? "yes" : "no"} ok={sys?.engine_present} />
      </section>

      {/* pipeline journey */}
      <section>
        <div className="eyebrow mb-3">The pipeline this lab automates</div>
        <ol className="flex flex-wrap gap-2">
          {PIPELINE.map((p) => (
            <a key={p.n} href={`#${p.to}`} className="card px-3 py-2 flex items-center gap-2 hover:bg-surface-2">
              <span className="mono text-accent tnum">{String(p.n).padStart(2, "0")}</span>
              <span className="text-sm">{p.label}</span>
            </a>
          ))}
        </ol>
        <p className="text-xs text-faint mt-3 mono">
          {err ? `backend: ${err}` : sys ? `python ${sys.python} · ${sys.cpus} CPUs · ${sys.data_dir}` : "loading…"}
        </p>
      </section>

      <section className="card p-5">
        <div className="eyebrow mb-2">Status</div>
        <pre className="mono text-xs text-ink-2 whitespace-pre-wrap">{JSON.stringify(health ?? {}, null, 2)}</pre>
      </section>
    </div>
  );
}

function Stat({ label, value, ok }: { label: string; value: string; ok?: boolean }) {
  return (
    <div className="card px-4 py-3">
      <div className="eyebrow">{label}</div>
      <div className={`mt-1 font-serif text-lg ${ok === false ? "text-danger" : ok ? "text-ok" : ""}`}>{value}</div>
    </div>
  );
}
