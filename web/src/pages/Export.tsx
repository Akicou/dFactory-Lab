import { useState } from "react";
import { useJobRunner, JobPill } from "../lib/hooks";

export default function Export() {
  const exp = useJobRunner();
  const [f, setF] = useState({ source: "", original_base_dir: "", export_name: "run1" });
  const set = (k: keyof typeof f, v: string) => setF((s) => ({ ...s, [k]: v }));

  return (
    <div className="space-y-9">
      <header>
        <div className="eyebrow">Phase 5 · Export</div>
        <h1 className="text-3xl mt-1">Export & packaging</h1>
        <p className="text-muted mt-2 max-w-2xl">Locate the trained checkpoint, split merged→separate, copy the modeling file, verify completeness, write a sha256 manifest.</p>
      </header>

      <section className="card p-5 space-y-3 max-w-2xl">
        <label className="block"><span className="eyebrow">Source (training output dir or checkpoint)</span>
          <input className="input mt-1 mono" value={f.source} onChange={(e) => set("source", e.target.value)} placeholder="…/llada2_mini_bd_sft_outputs" /></label>
        <label className="block"><span className="eyebrow">Original base model dir (for modeling file)</span>
          <input className="input mt-1 mono" value={f.original_base_dir} onChange={(e) => set("original_base_dir", e.target.value)} placeholder="…/LLaDA2.0-mini-preview" /></label>
        <label className="block"><span className="eyebrow">Export name</span>
          <input className="input mt-1" value={f.export_name} onChange={(e) => set("export_name", e.target.value)} /></label>
        <button className="btn btn-primary" disabled={exp.running || !f.source || !f.original_base_dir}
          onClick={() => exp.run("/api/export", f)}>
          {exp.running ? "exporting…" : "Export runnable model"}
        </button>
        <div className="flex items-center gap-2 text-sm">
          {exp.job && <><JobPill state={exp.job.state} /><span className="text-muted">{exp.job.message}</span></>}
          {exp.error && <span className="text-danger mono text-xs">{exp.error.slice(0, 160)}</span>}
        </div>
        {exp.job?.result ? (
          <pre className="mono text-xs bg-surface-2 p-3 rounded whitespace-pre-wrap overflow-auto">
            {JSON.stringify(exp.job.result, null, 2)}
          </pre>
        ) : null}
      </section>
    </div>
  );
}
