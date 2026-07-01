import { useState, type ReactNode } from "react";
import { FlaskConical, Play, Sparkles } from "lucide-react";
import { type LocalModel, type DatasetRow } from "../lib/api";
import { useApi, useJobRunner, JobPill } from "../lib/hooks";
import { Page } from "../components/Page";

export default function Training() {
  const models = useApi<LocalModel[]>("/api/models/local", 5000);
  const datasets = useApi<DatasetRow[]>("/api/datasets", 5000);
  const run = useJobRunner();

  const [modelSource, setModelSource] = useState("");
  const [datasetPath, setDatasetPath] = useState("");
  const [preset, setPreset] = useState("llada2-mini");
  const [f, setF] = useState({ lr: "1.0e-5", global_batch_size: "8", num_train_epochs: "1", block_size: "32", noise_range_low: "0.3", noise_range_high: "0.8" });
  const [dryRun, setDryRun] = useState(true);
  const set = (k: keyof typeof f, v: string) => setF((s) => ({ ...s, [k]: v }));

  const overrides = () => ({
    train: { lr: Number(f.lr), global_batch_size: Number(f.global_batch_size),
             num_train_epochs: Number(f.num_train_epochs), block_size: Number(f.block_size) },
    data: { noise_range_low: Number(f.noise_range_low), noise_range_high: Number(f.noise_range_high) },
  });

  const start = () => run.run("/api/training/finetune", {
    model_source: modelSource, dataset_path: datasetPath, preset, overrides: overrides(), dry_run: dryRun,
  });

  const jobResult = run.job?.result as { merged?: boolean; model_path?: string; launch?: { argv?: string[]; command?: string } } | undefined;

  return (
    <Page title="Training" icon={FlaskConical}>
      <div className="grid lg:grid-cols-2 gap-3 anim-stagger">
        <Field label="Model source">
          <select className="input" value={modelSource} onChange={(e) => setModelSource(e.target.value)}>
            <option value="">Select a local model…</option>
            {(models.data ?? []).map((m) => <option key={m.id} value={m.path}>{m.id} · {m.format}</option>)}
          </select>
          <input className="input mt-2 mono text-xs" placeholder="…or paste a model dir" value={modelSource} onChange={(e) => setModelSource(e.target.value)} />
        </Field>
        <Field label="Dataset">
          <select className="input" value={datasetPath} onChange={(e) => setDatasetPath(e.target.value)}>
            <option value="">Select a dataset…</option>
            {(datasets.data ?? []).map((d) => <option key={d.id} value={d.path + "/train.jsonl"}>{d.name}</option>)}
          </select>
          <input className="input mt-2 mono text-xs" placeholder="…or paste a train.jsonl path" value={datasetPath} onChange={(e) => setDatasetPath(e.target.value)} />
        </Field>
      </div>

      <div className="card p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="label">Hyperparameters</div>
          <div className="flex items-center gap-2">
            <select className="input w-40" value={preset} onChange={(e) => setPreset(e.target.value)}>
              <option value="llada2-mini">llada2-mini</option>
              <option value="llada2-flash">llada2-flash</option>
              <option value="llada2-21-mini">llada2-21-mini</option>
              <option value="llada2-21-flash">llada2-21-flash</option>
              <option value="llada2-21-mini-256k">llada2-21-mini-256k</option>
            </select>
          </div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <NumField label="lr" value={f.lr} onChange={(v) => set("lr", v)} />
          <NumField label="global_batch_size" value={f.global_batch_size} onChange={(v) => set("global_batch_size", v)} />
          <NumField label="num_train_epochs" value={f.num_train_epochs} onChange={(v) => set("num_train_epochs", v)} />
          <NumField label="block_size" value={f.block_size} onChange={(v) => set("block_size", v)} accent />
          <NumField label="noise_range_low" value={f.noise_range_low} onChange={(v) => set("noise_range_low", v)} accent />
          <NumField label="noise_range_high" value={f.noise_range_high} onChange={(v) => set("noise_range_high", v)} accent />
        </div>
      </div>

      <div className="card p-5">
        <div className="flex flex-wrap items-center gap-3">
          <button className="btn btn-primary" disabled={run.running || !modelSource || !datasetPath} onClick={start}>
            <Play size={15} /> {run.running ? "Running…" : "Start finetune"}
          </button>
          <label className="flex items-center gap-2 text-sm text-muted cursor-pointer">
            <input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} /> dry run
          </label>
          <span className="text-xs text-faint">The model is merged automatically before training.</span>
          {run.job && <div className="flex items-center gap-2"><JobPill state={run.job.state} /><span className="text-sm text-muted">{run.job.message}</span></div>}
        </div>
        {run.error && <div className="mt-3 text-danger mono text-xs whitespace-pre-wrap">{run.error.slice(0, 400)}</div>}
        {jobResult?.launch?.command && (
          <div className="mt-4">
            {jobResult.merged && <div className="text-xs text-ok mb-2 flex items-center gap-1"><Sparkles size={12} /> merged model for training</div>}
            <div className="label mb-1">torchrun command</div>
            <pre className="mono text-xs bg-sunken rounded-lg p-3 overflow-auto whitespace-pre-wrap">{jobResult.launch.command}</pre>
          </div>
        )}
      </div>
    </Page>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return <div><div className="label mb-1.5">{label}</div>{children}</div>;
}
function NumField({ label, value, onChange, accent }: { label: string; value: string; onChange: (v: string) => void; accent?: boolean }) {
  return (
    <label className="block">
      <span className={`label ${accent ? "text-ink" : ""}`}>{label}</span>
      <input className={`input mt-1 mono ${accent ? "border-ink/30" : ""}`} value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}
