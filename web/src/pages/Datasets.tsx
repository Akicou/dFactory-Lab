import { useState } from "react";
import { api, type DatasetRow } from "../lib/api";
import { useApi, useJobRunner, JobPill } from "../lib/hooks";

const SAMPLE = JSON.stringify(
  [{ question: "What is 2+2?", answer: "4" }, { question: "Capital of France?", answer: "Paris" }], null, 2);

export default function Datasets() {
  const list = useApi<DatasetRow[]>("/api/datasets", 3000);
  const build = useJobRunner();
  const [rows, setRows] = useState(SAMPLE);
  const [name, setName] = useState("my-dataset");
  const [userCol, setUserCol] = useState("question");
  const [asstCol, setAsstCol] = useState("answer");
  const [conv, setConv] = useState<{ id?: string; stats?: Record<string, number>; invalid?: number; error?: string } | null>(null);
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [preview, setPreview] = useState<unknown>(null);

  const convert = async () => {
    setConv(null);
    try {
      const parsed = JSON.parse(rows);
      const res = await api<{ id: string; stats: Record<string, number>; invalid_count: number }>(
        "/api/datasets/convert",
        { method: "POST", body: JSON.stringify({ rows: parsed, name, mapping: { [userCol]: "user", [asstCol]: "assistant" } }) });
      setConv({ id: res.id, stats: res.stats, invalid: res.invalid_count });
      list.reload();
    } catch (e) { setConv({ error: String(e instanceof Error ? e.message : e) }); }
  };

  const openPreview = async (id: string) => {
    setPreviewId(id);
    try { setPreview(await api(`/api/datasets/${id}/preview`)); } catch (e) { setPreview({ error: String(e instanceof Error ? e.message : e) }); }
  };

  return (
    <div className="space-y-9">
      <header>
        <div className="eyebrow">Phase 3 · Datasets</div>
        <h1 className="text-3xl mt-1">Datasets & preparation</h1>
        <p className="text-muted mt-2 max-w-2xl">Convert rows to conversational messages JSONL, validate, preview, or build the GSM8K preset.</p>
      </header>

      <section className="grid lg:grid-cols-2 gap-6">
        <div className="card p-4 space-y-3">
          <div className="eyebrow">Convert rows → messages</div>
          <div className="grid grid-cols-3 gap-2">
            <input className="input col-span-1" placeholder="name" value={name} onChange={(e) => setName(e.target.value)} />
            <input className="input" placeholder="user col" value={userCol} onChange={(e) => setUserCol(e.target.value)} />
            <input className="input" placeholder="assistant col" value={asstCol} onChange={(e) => setAsstCol(e.target.value)} />
          </div>
          <textarea className="input mono text-xs h-40" value={rows} onChange={(e) => setRows(e.target.value)} />
          <button className="btn btn-primary" onClick={convert}>Convert &amp; save</button>
          {conv?.error && <div className="text-danger text-xs mono">{conv.error}</div>}
          {conv?.stats && (
            <div className="text-xs text-muted">
              saved <span className="mono">{conv.id}</span> · {conv.stats.rows} rows · {conv.stats.turns} turns · {conv.invalid} invalid
            </div>
          )}
        </div>

        <div className="card p-4 space-y-3">
          <div className="eyebrow">Build preset</div>
          <p className="text-sm text-muted">openai/gsm8k → conversational JSONL (question→user, answer→assistant).</p>
          <button className="btn btn-primary" disabled={build.running}
            onClick={() => build.run("/api/datasets/build", { preset: "gsm8k", name: "gsm8k" })}>
            {build.running ? "building…" : "Build GSM8K"}
          </button>
          <div className="text-sm flex items-center gap-2">
            {build.job && <><JobPill state={build.job.state} /><span className="text-muted">{build.job.message}</span></>}
            {build.error && <span className="text-danger mono text-xs">{build.error.slice(0, 120)}</span>}
          </div>
        </div>
      </section>

      <section>
        <div className="eyebrow mb-3">Saved datasets</div>
        <div className="card divide-y rule">
          {(list.data ?? []).map((d) => (
            <div key={d.id} className="flex items-center gap-3 px-4 py-2.5">
              <div className="flex-1 min-w-0">
                <div className="font-medium">{d.name}</div>
                <div className="mono text-xs text-faint truncate">{d.path}</div>
              </div>
              <span className="mono text-xs text-muted tnum">{d.rows} rows</span>
              <span className="badge mono">{d.source}</span>
              <button className="btn" onClick={() => openPreview(d.id)}>preview</button>
            </div>
          ))}
          {(!list.data || list.data.length === 0) && <div className="px-4 py-8 text-center text-faint text-sm">No datasets yet.</div>}
        </div>
      </section>

      {previewId && (
        <section className="card p-4">
          <div className="eyebrow mb-2">Preview · {previewId}</div>
          <pre className="mono text-xs whitespace-pre-wrap max-h-72 overflow-auto">{JSON.stringify(preview, null, 2)}</pre>
        </section>
      )}
    </div>
  );
}
