import { useState } from "react";
import { type ModelMeta, type LocalModel } from "@/lib/api";
import { useApi, useJobRunner, JobPill, fmtBytes } from "@/lib/hooks";

export default function Models() {
  const catalog = useApi<ModelMeta[]>("/api/models", 0);
  const local = useApi<LocalModel[]>("/api/models/local", 4000);
  const dl = useJobRunner();
  const merge = useJobRunner();
  const [mergeInput, setMergeInput] = useState("");
  const [mergeOut, setMergeOut] = useState("");

  return (
    <div className="space-y-9">
      <header>
        <div className="eyebrow">Phase 2 · Model hub</div>
        <h1 className="text-3xl mt-1">Models & MoE merge / split</h1>
        <p className="text-muted mt-2 max-w-2xl">Download a dLLM, merge separate→merged experts for training, split back after — all as managed jobs.</p>
      </header>

      {/* catalog */}
      <section>
        <div className="eyebrow mb-3">Supported dLLMs</div>
        <div className="grid md:grid-cols-2 gap-3">
          {(catalog.data ?? []).map((m) => (
            <div key={m.id} className="card p-4">
              <div className="flex items-baseline justify-between">
                <h3 className="text-lg">{m.name}</h3>
                <span className="badge mono">{m.size}</span>
              </div>
              <div className="mono text-xs text-muted mt-1">{m.repo_id}</div>
              <dl className="grid grid-cols-3 gap-2 mt-3 text-xs">
                {[["experts", m.meta.num_experts], ["layers", m.meta.num_hidden_layers],
                  ["hidden", m.meta.hidden_size], ["per-tok", m.meta.num_experts_per_tok],
                  ["vocab", m.meta.vocab_size], ["pad_id", m.meta.pad_token_id]].map(([k, v]) => (
                  <div key={String(k)}><dt className="eyebrow">{k}</dt><dd className="mono tnum">{String(v)}</dd></div>
                ))}
              </dl>
              <button className="btn btn-primary mt-3" disabled={dl.running}
                onClick={() => dl.run("/api/models/download", { repo_id: m.repo_id })}>
                {dl.running ? "downloading…" : "Download"}
              </button>
            </div>
          ))}
        </div>
        <JobLine label="download" runner={dl} />
      </section>

      {/* inventory */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <div className="eyebrow">Local models</div>
          <button className="btn" onClick={local.reload}>refresh</button>
        </div>
        <div className="card divide-y rule">
          {(local.data ?? []).map((m) => (
            <div key={m.id} className="flex items-center gap-4 px-4 py-3">
              <div className="flex-1 min-w-0">
                <div className="font-medium truncate">{m.id}</div>
                <div className="mono text-xs text-faint truncate">{m.path}</div>
              </div>
              <span className="badge mono">{m.format}</span>
              <span className="mono text-xs text-muted tnum w-20 text-right">{fmtBytes(m.size_bytes)}</span>
              {m.format === "separate_expert" && (
                <button className="btn" onClick={() => merge.run("/api/models/merge",
                  { input_dir: m.path, output_dir: m.path + "-merged" })}>merge</button>
              )}
            </div>
          ))}
          {(!local.data || local.data.length === 0) && (
            <div className="px-4 py-8 text-center text-faint text-sm">No local models yet — download one above.</div>
          )}
        </div>
      </section>

      {/* manual convert */}
      <section className="card p-4 space-y-3">
        <div className="eyebrow">Manual merge</div>
        <div className="grid md:grid-cols-2 gap-3">
          <input className="input" placeholder="input dir (separate-expert)" value={mergeInput} onChange={(e) => setMergeInput(e.target.value)} />
          <input className="input" placeholder="output dir (merged)" value={mergeOut} onChange={(e) => setMergeOut(e.target.value)} />
        </div>
        <button className="btn btn-primary" disabled={merge.running || !mergeInput || !mergeOut}
          onClick={() => merge.run("/api/models/merge", { input_dir: mergeInput, output_dir: mergeOut })}>
          {merge.running ? "merging…" : "Merge experts"}
        </button>
        <JobLine label="merge" runner={merge} />
      </section>
    </div>
  );
}

function JobLine({ label, runner }: { label: string; runner: ReturnType<typeof useJobRunner> }) {
  if (!runner.job && !runner.running && !runner.error) return null;
  return (
    <div className="mt-2 text-sm flex items-center gap-2">
      <span className="eyebrow">{label}</span>
      {runner.job && <JobPill state={runner.job.state} />}
      <span className="text-muted">{runner.job?.message}</span>
      {runner.error && <span className="text-danger mono text-xs">{runner.error.slice(0, 160)}</span>}
      {runner.job?.result ? <span className="mono text-xs text-faint">{JSON.stringify(runner.job.result).slice(0, 120)}</span> : null}
    </div>
  );
}
