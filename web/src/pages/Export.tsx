import { useState } from "react";
import { Save, FolderDown } from "lucide-react";
import { useJobRunner, JobPill } from "../lib/hooks";
import { Page } from "../components/Page";

export default function Export() {
  const exp = useJobRunner();
  const [f, setF] = useState({ source: "", original_base_dir: "", export_name: "run1" });
  const set = (k: keyof typeof f, v: string) => setF((s) => ({ ...s, [k]: v }));

  const res = exp.job?.result as { export_dir?: string; modeling_files_copied?: string[]; missing?: string[] } | undefined;

  return (
    <Page title="Save model" icon={Save}>
      <div className="card p-5 max-w-2xl space-y-4 anim-rise">
        <p className="text-sm text-muted">Runs the standard post-training steps: split merged experts, copy the modeling file, verify, and write a manifest.</p>
        <label className="block">
          <span className="label">Training output dir</span>
          <input className="input mt-1.5 mono text-xs" value={f.source} onChange={(e) => set("source", e.target.value)} placeholder="…/outputs/mini" />
        </label>
        <label className="block">
          <span className="label">Original base model dir</span>
          <input className="input mt-1.5 mono text-xs" value={f.original_base_dir} onChange={(e) => set("original_base_dir", e.target.value)} placeholder="…/LLaDA2.0-mini-preview" />
        </label>
        <label className="block">
          <span className="label">Export name</span>
          <input className="input mt-1.5" value={f.export_name} onChange={(e) => set("export_name", e.target.value)} />
        </label>
        <button className="btn btn-primary" disabled={exp.running || !f.source || !f.original_base_dir}
          onClick={() => exp.run("/api/export", f)}>
          <Save size={15} /> {exp.running ? "Saving…" : "Save model"}
        </button>
        <div className="flex items-center gap-2">
          {exp.job && <><JobPill state={exp.job.state} /><span className="text-sm text-muted">{exp.job.message}</span></>}
          {exp.error && <span className="text-danger mono text-xs">{exp.error.slice(0, 160)}</span>}
        </div>
        {res && (
          <div className="rounded-lg bg-sunken p-3 text-xs space-y-1">
            <div className="flex items-center gap-1.5 text-ok"><FolderDown size={13} /> <span className="mono">{res.export_dir}</span></div>
            {res.modeling_files_copied && <div className="text-muted">copied: {res.modeling_files_copied.join(", ")}</div>}
            {res.missing && res.missing.length > 0 && <div className="text-warn">missing: {res.missing.join(", ")}</div>}
          </div>
        )}
      </div>
    </Page>
  );
}
