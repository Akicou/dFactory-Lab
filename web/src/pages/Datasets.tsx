import { useState } from "react";
import { Database, Plus, Eye, Hammer } from "lucide-react";
import { api, type DatasetRow } from "../lib/api";
import { useApi, useJobRunner, JobPill } from "../lib/hooks";
import { Page } from "../components/Page";

const SAMPLE = JSON.stringify(
  [{ question: "What is 2+2?", answer: "4" }, { question: "Capital of France?", answer: "Paris" }], null, 2);

export default function Datasets() {
  const list = useApi<DatasetRow[]>("/api/datasets", 3000);
  const build = useJobRunner();
  const [rows, setRows] = useState(SAMPLE);
  const [name, setName] = useState("my-dataset");
  const [userCol, setUserCol] = useState("question");
  const [asstCol, setAsstCol] = useState("answer");
  const [conv, setConv] = useState<{ id?: string; rows?: number; invalid?: number; error?: string } | null>(null);
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [preview, setPreview] = useState<{ records?: unknown[]; stats?: { rows: number } } | null>(null);

  const convert = async () => {
    setConv(null);
    try {
      const parsed = JSON.parse(rows);
      const res = await api<{ id: string; stats: { rows: number }; invalid_count: number }>("/api/datasets/convert",
        { method: "POST", body: JSON.stringify({ rows: parsed, name, mapping: { [userCol]: "user", [asstCol]: "assistant" } }) });
      setConv({ id: res.id, rows: res.stats.rows, invalid: res.invalid_count });
      list.reload();
    } catch (e) { setConv({ error: String(e instanceof Error ? e.message : e) }); }
  };

  const openPreview = async (id: string) => {
    setPreviewId(id);
    try { setPreview(await api(`/api/datasets/${id}/preview`)); } catch (e) { setPreview({}); }
  };

  return (
    <Page title="Datasets" icon={Database}>
      <div className="grid lg:grid-cols-2 gap-3 anim-stagger">
        <div className="card p-5">
          <div className="label mb-3">Convert rows to messages</div>
          <div className="grid grid-cols-3 gap-2 mb-2">
            <input className="input" placeholder="name" value={name} onChange={(e) => setName(e.target.value)} />
            <input className="input" placeholder="user col" value={userCol} onChange={(e) => setUserCol(e.target.value)} />
            <input className="input" placeholder="assistant col" value={asstCol} onChange={(e) => setAsstCol(e.target.value)} />
          </div>
          <textarea className="input mono text-xs h-36" value={rows} onChange={(e) => setRows(e.target.value)} />
          <button className="btn btn-primary mt-3" onClick={convert}><Plus size={15} /> Convert &amp; save</button>
          {conv?.error && <div className="text-danger text-xs mono mt-2">{conv.error}</div>}
          {conv?.id && <div className="text-xs text-muted mt-2">saved <span className="mono">{conv.id}</span> · {conv.rows} rows{conv.invalid ? ` · ${conv.invalid} invalid` : ""}</div>}
        </div>

        <div className="card p-5">
          <div className="label mb-3">Preset</div>
          <p className="text-sm text-muted">openai/gsm8k to conversational JSONL.</p>
          <button className="btn btn-primary mt-3" disabled={build.running}
            onClick={() => build.run("/api/datasets/build", { preset: "gsm8k", name: "gsm8k" })}>
            <Hammer size={15} /> {build.running ? "Building…" : "Build GSM8K"}
          </button>
          <div className="mt-3 flex items-center gap-2">
            {build.job && <><JobPill state={build.job.state} /><span className="text-sm text-muted">{build.job.message}</span></>}
            {build.error && <span className="text-danger mono text-xs">{build.error.slice(0, 120)}</span>}
          </div>
        </div>
      </div>

      <div>
        <div className="label mb-3">Saved</div>
        <div className="card divide-y divide-edge">
          {(list.data ?? []).map((d) => (
            <div key={d.id} className="flex items-center gap-3 px-4 py-2.5">
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium">{d.name}</div>
                <div className="mono text-xs text-faint truncate">{d.path}</div>
              </div>
              <span className="mono text-xs text-muted tnum">{d.rows} rows</span>
              <span className="badge">{d.source}</span>
              <button className="btn btn-ghost" onClick={() => openPreview(d.id)}><Eye size={14} /></button>
            </div>
          ))}
          {(!list.data || list.data.length === 0) && <div className="px-4 py-10 text-center text-sm text-faint">No datasets yet.</div>}
        </div>
      </div>

      {previewId && (
        <div className="card p-5">
          <div className="label mb-2">Preview · {previewId}</div>
          <pre className="mono text-xs whitespace-pre-wrap max-h-72 overflow-auto">{JSON.stringify(preview, null, 2)}</pre>
        </div>
      )}
    </Page>
  );
}
