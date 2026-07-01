import { Boxes, Download, RefreshCw } from "lucide-react";
import { type ModelMeta, type LocalModel } from "../lib/api";
import { useApi, useJobRunner, JobPill, fmtBytes } from "../lib/hooks";
import { Page } from "../components/Page";

export default function Models() {
  const catalog = useApi<ModelMeta[]>("/api/models", 0);
  const local = useApi<LocalModel[]>("/api/models/local", 4000);
  const dl = useJobRunner();

  return (
    <Page title="Models" icon={Boxes}
      actions={<button className="btn btn-ghost" onClick={local.reload}><RefreshCw size={15} /> Refresh</button>}>
      <div className="grid md:grid-cols-2 gap-3 anim-stagger">
        {(catalog.data ?? []).map((m) => (
          <div key={m.id} className="card p-4 card-hover">
            <div className="flex items-baseline justify-between">
              <h3 className="text-base font-semibold">{m.name}</h3>
              <span className="badge">{m.size}</span>
            </div>
            <div className="mono text-xs text-faint mt-1 truncate">{m.repo_id}</div>
            <div className="grid grid-cols-3 gap-2 mt-3">
              {[["experts", m.meta.num_experts], ["layers", m.meta.num_hidden_layers], ["hidden", m.meta.hidden_size]].map(([k, v]) => (
                <div key={String(k)}>
                  <div className="label">{String(k)}</div>
                  <div className="mono text-sm tnum">{String(v)}</div>
                </div>
              ))}
            </div>
            <button className="btn btn-primary mt-4 w-full" disabled={dl.running}
              onClick={() => dl.run("/api/models/download", { repo_id: m.repo_id })}>
              <Download size={15} /> {dl.running ? "Downloading…" : "Download"}
            </button>
          </div>
        ))}
      </div>

      <div>
        <div className="label mb-3">Local models</div>
        <div className="card divide-y divide-edge">
          {(local.data ?? []).map((m) => (
            <div key={m.id} className="flex items-center gap-4 px-4 py-3">
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate">{m.id}</div>
                <div className="mono text-xs text-faint truncate">{m.path}</div>
              </div>
              <span className="badge">{m.format}</span>
              <span className="mono text-xs text-muted tnum w-20 text-right">{fmtBytes(m.size_bytes)}</span>
            </div>
          ))}
          {(!local.data || local.data.length === 0) && (
            <div className="px-4 py-10 text-center text-sm text-faint">No local models. Download one above.</div>
          )}
        </div>
        <p className="text-xs text-faint mt-3">
          Merging happens automatically when you start a finetune in Training.
        </p>
        {dl.job && (
          <div className="mt-3 text-sm flex items-center gap-2">
            <JobPill state={dl.job.state} />
            <span className="text-muted">{dl.job.message}</span>
            {dl.error && <span className="text-danger mono text-xs">{dl.error.slice(0, 140)}</span>}
          </div>
        )}
      </div>
    </Page>
  );
}
