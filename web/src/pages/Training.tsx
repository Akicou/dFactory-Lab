import { useState } from "react";
import { api } from "@/lib/api";
import { useApi, useJobRunner, JobPill } from "@/lib/hooks";

interface Run { id: string; status: string; config_path: string; output_dir: string; created_at: string }

export default function Training() {
  const cfg = useApi<{ presets: string[]; diffusion_keys: string[] }>("/api/training/config", 0);
  const runs = useApi<Run[]>("/api/training/runs", 3000);
  const launch = useJobRunner();
  const [preset, setPreset] = useState("llada2-mini");
  const [f, setF] = useState({
    model_path: "./LLaDA2.0-mini-preview-moe-merge", train_path: "./gsm8k_datasets/gsm8k_train.jsonl",
    output_dir: "./outputs/mini", lr: "1.0e-5", global_batch_size: "8", num_train_epochs: "1",
    max_seq_len: "2048", noise_range_low: "0.3", noise_range_high: "0.8", block_size: "32",
    block_diffusion_mode: true,
  });
  const [validation, setValidation] = useState<{ valid: boolean; errors: string[] } | null>(null);

  const overrides = () => ({
    model: { model_path: f.model_path },
    data: { train_path: f.train_path, max_seq_len: Number(f.max_seq_len),
            noise_range_low: Number(f.noise_range_low), noise_range_high: Number(f.noise_range_high) },
    train: { output_dir: f.output_dir, lr: Number(f.lr), global_batch_size: Number(f.global_batch_size),
             num_train_epochs: Number(f.num_train_epochs), block_size: Number(f.block_size),
             block_diffusion_mode: f.block_diffusion_mode },
  });

  const validate = async () => {
    try { setValidation(await api("/api/training/config/validate", { method: "POST", body: JSON.stringify({ config: overrides() }) })); }
    catch (e) { setValidation({ valid: false, errors: [String(e instanceof Error ? e.message : e)] }); }
  };

  const start = () => launch.run("/api/training/start", { preset, overrides: overrides(), dry_run: true });
  const set = (k: keyof typeof f, v: string | boolean) => setF((s) => ({ ...s, [k]: v }));

  return (
    <div className="space-y-9">
      <header>
        <div className="eyebrow">Phase 4 · Training</div>
        <h1 className="text-3xl mt-1">Block-diffusion SFT</h1>
        <p className="text-muted mt-2 max-w-2xl">Configure every YAML key — including the diffusion knobs — then launch torchrun (dry-run here; real run needs GPU + VeOmni).</p>
      </header>

      <section className="card p-5 space-y-4">
        <div className="flex items-center gap-3">
          <span className="eyebrow">Preset</span>
          <select className="input w-48" value={preset} onChange={(e) => setPreset(e.target.value)}>
            {(cfg.data?.presets ?? ["llada2-mini", "llada2-flash"]).map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
        <div className="grid md:grid-cols-3 gap-3">
          <Field label="model_path" value={f.model_path} onChange={(v) => set("model_path", v)} />
          <Field label="train_path" value={f.train_path} onChange={(v) => set("train_path", v)} />
          <Field label="output_dir" value={f.output_dir} onChange={(v) => set("output_dir", v)} />
          <Field label="lr" value={f.lr} onChange={(v) => set("lr", v)} mono />
          <Field label="global_batch_size" value={f.global_batch_size} onChange={(v) => set("global_batch_size", v)} mono />
          <Field label="num_train_epochs" value={f.num_train_epochs} onChange={(v) => set("num_train_epochs", v)} mono />
          <Field label="max_seq_len" value={f.max_seq_len} onChange={(v) => set("max_seq_len", v)} mono />
          <Field label="noise_range_low" value={f.noise_range_low} onChange={(v) => set("noise_range_low", v)} mono accent />
          <Field label="noise_range_high" value={f.noise_range_high} onChange={(v) => set("noise_range_high", v)} mono accent />
          <Field label="block_size" value={f.block_size} onChange={(v) => set("block_size", v)} mono accent />
          <label className="flex items-center gap-2 text-sm self-end pb-2">
            <input type="checkbox" checked={f.block_diffusion_mode} onChange={(e) => set("block_diffusion_mode", e.target.checked)} />
            block_diffusion_mode
          </label>
        </div>
        <div className="flex items-center gap-3">
          <button className="btn" onClick={validate}>Validate</button>
          <button className="btn btn-primary" disabled={launch.running} onClick={start}>
            {launch.running ? "launching…" : "Launch (dry-run)"}
          </button>
          {validation && (
            <span className={`text-sm ${validation.valid ? "text-ok" : "text-danger"}`}>
              {validation.valid ? "config valid ✓" : validation.errors.join("; ")}
            </span>
          )}
        </div>
        {launch.job?.result ? (
          <pre className="mono text-xs bg-surface-2 p-3 rounded whitespace-pre-wrap overflow-auto">
            {(launch.job.result as { argv?: string[] }).argv?.join(" ")}
          </pre>
        ) : null}
        {launch.error && <div className="text-danger mono text-xs">{launch.error}</div>}
      </section>

      <section>
        <div className="eyebrow mb-3">Training runs</div>
        <div className="card divide-y rule">
          {(runs.data ?? []).map((r) => (
            <div key={r.id} className="flex items-center gap-3 px-4 py-2.5">
              <span className="mono text-xs text-faint w-28 truncate">{r.id}</span>
              <JobPill state={r.status} />
              <span className="mono text-xs text-muted flex-1 truncate">{r.output_dir}</span>
            </div>
          ))}
          {(!runs.data || runs.data.length === 0) && <div className="px-4 py-8 text-center text-faint text-sm">No runs yet.</div>}
        </div>
      </section>
    </div>
  );
}

function Field({ label, value, onChange, mono, accent }: { label: string; value: string; onChange: (v: string) => void; mono?: boolean; accent?: boolean }) {
  return (
    <label className="block">
      <span className={`eyebrow ${accent ? "text-accent" : ""}`}>{label}</span>
      <input className={`input mt-1 ${mono ? "mono" : ""}`} value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}
