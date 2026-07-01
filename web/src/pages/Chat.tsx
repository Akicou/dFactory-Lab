import { useState } from "react";
import { MessageSquare, Send, Sparkles, X, Plus, Columns2 } from "lucide-react";
import { api, type LocalModel, type LoadedServer } from "../lib/api";
import { useApi } from "../lib/hooks";
import { Page } from "../components/Page";

interface GenResult { response: string; details: { diffusion_steps: number; unmasking: { step: number; unmasked_frac: number }[] | null; tokens_generated: number } }

export default function Chat() {
  const models = useApi<LocalModel[]>("/api/models/local", 4000);
  const loaded = useApi<LoadedServer[]>("/api/chat/loaded", 1500);
  const servers = loaded.data ?? [];
  const ready = servers.filter((s) => s.state === "ready");

  const [pick, setPick] = useState("");        // model dir to load
  const [server, setServer] = useState("");     // active server for single chat
  const [leftId, setLeftId] = useState("");
  const [rightId, setRightId] = useState("");
  const [compare, setCompare] = useState(false);

  const [msgs, setMsgs] = useState<{ role: string; content: string }[]>([{ role: "user", content: "Explain masked diffusion in one line." }]);
  const [rounds, setRounds] = useState<{ prompt: string; left: string; right: string }[]>([]);
  const [input, setInput] = useState("");
  const [steps, setSteps] = useState(32);
  const [schedule, setSchedule] = useState("linear");
  const [showUnmask, setShowUnmask] = useState(true);
  const [res, setRes] = useState<GenResult | null>(null);
  const [busy, setBusy] = useState(false);

  const params = () => ({ diffusion_steps: steps, mask_schedule: schedule, show_unmasking: showUnmask });

  const load = async () => {
    if (!pick) return;
    try { await api("/api/chat/load", { method: "POST", body: JSON.stringify({ model_path: pick }) }); }
    catch (e) { alert(String(e instanceof Error ? e.message : e)); }
    loaded.reload();
  };
  const eject = async (id: string) => {
    await api("/api/chat/eject", { method: "POST", body: JSON.stringify({ id }) }).catch(() => {});
    loaded.reload();
  };

  const send = async () => {
    if (!input.trim() || busy) return;
    const prompt = input;
    setBusy(true); setInput("");
    try {
      if (compare) {
        const r = await api<{ left: GenResult; right: GenResult }>("/api/chat/compare", {
          method: "POST",
          body: JSON.stringify({ left_id: leftId, right_id: rightId, messages: [...msgs, { role: "user", content: prompt }], params: params() }),
        });
        setRounds((rs) => [...rs, { prompt, left: r.left.response, right: r.right.response }]);
      } else {
        const next = [...msgs, { role: "user", content: prompt }];
        setMsgs(next);
        const r = await api<GenResult>("/api/chat/completions", {
          method: "POST",
          body: JSON.stringify({ server_id: server || undefined, messages: next, params: params() }),
        });
        setRes(r); setMsgs([...next, { role: "assistant", content: r.response }]);
      }
    } catch (e) { alert(String(e instanceof Error ? e.message : e)); } finally { setBusy(false); }
  };

  return (
    <Page title="Chat" icon={MessageSquare}
      actions={
        <button className={`btn ${compare ? "btn-primary" : "btn-ghost"}`} onClick={() => setCompare((c) => !c)}>
          <Columns2 size={15} /> Compare
        </button>
      }>
      {/* model bar: load + loaded chips */}
      <div className="card p-3 flex flex-wrap items-center gap-2">
        <select className="input w-56" value={pick} onChange={(e) => setPick(e.target.value)}>
          <option value="">Select a local model…</option>
          {(models.data ?? []).map((m) => <option key={m.id} value={m.path}>{m.id}</option>)}
        </select>
        <button className="btn btn-primary" onClick={load} disabled={!pick}><Plus size={15} /> Load</button>
        <div className="flex flex-wrap gap-2 ml-auto">
          {servers.map((s) => (
            <span key={s.id} className={`badge flex items-center gap-1.5 ${s.state === "ready" ? "text-ok" : s.state === "error" ? "text-danger" : "text-warn"}`}
              title={s.message || s.model_path}>
              {s.state === "starting" && <span className="pulse">◐</span>}
              {s.name} · {s.state}
              <button onClick={() => eject(s.id)} className="hover:text-danger"><X size={12} /></button>
            </span>
          ))}
          {servers.length === 0 && <span className="text-xs text-faint">No models loaded — using mock backend.</span>}
        </div>
      </div>

      <div className="grid lg:grid-cols-[1fr_18rem] gap-3 mt-3">
        {compare ? (
          <ArenaPanes rounds={rounds} busy={busy} ready={ready}
            leftId={leftId} rightId={rightId} setLeftId={setLeftId} setRightId={setRightId} />
        ) : (
          <div className="card p-4 flex flex-col h-[30rem] anim-rise">
            <div className="mb-2">
              <select className="input w-64" value={server} onChange={(e) => setServer(e.target.value)}>
                <option value="">mock backend</option>
                {ready.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
              </select>
            </div>
            <div className="flex-1 overflow-auto space-y-3 pr-1">
              {msgs.map((m, i) => (
                <div key={i} className={`anim-fade ${m.role === "user" ? "text-right" : ""}`}>
                  <div className={`inline-block max-w-[85%] px-3.5 py-2 rounded-lg text-sm ${m.role === "user" ? "bg-primary text-primary-fg" : "bg-sunken"}`}>{m.content}</div>
                </div>
              ))}
              {busy && <div className="text-faint text-sm pulse">denoising…</div>}
            </div>
          </div>
        )}

        <aside className="space-y-3">
          <div className="card p-4 space-y-3">
            <div className="label">Diffusion</div>
            <div>
              <div className="flex justify-between text-xs mb-1"><span className="label">steps</span><span className="mono tnum">{steps}</span></div>
              <input type="range" min={1} max={256} value={steps} onChange={(e) => setSteps(Number(e.target.value))} className="w-full accent-zinc-900" />
            </div>
            <div>
              <span className="label">mask schedule</span>
              <select className="input mt-1" value={schedule} onChange={(e) => setSchedule(e.target.value)}>
                <option>linear</option><option>geometric</option><option>cosine</option>
              </select>
            </div>
            <label className="flex items-center gap-2 text-sm text-muted">
              <input type="checkbox" checked={showUnmask} onChange={(e) => setShowUnmask(e.target.checked)} /> show unmasking
            </label>
          </div>

          {!compare && res?.details.unmasking && (
            <div className="card p-4 anim-rise">
              <div className="label mb-2 flex items-center gap-1"><Sparkles size={12} /> Unmasking · {res.details.diffusion_steps}</div>
              <div className="flex items-end gap-px h-20">
                {res.details.unmasking.map((s) => (
                  <div key={s.step} className="flex-1 bg-primary rounded-t-sm anim-rise"
                    style={{ height: `${s.unmasked_frac * 100}%`, animationDelay: `${s.step * 8}ms` }}
                    title={`step ${s.step + 1}: ${Math.round(s.unmasked_frac * 100)}%`} />
                ))}
              </div>
              <div className="text-xs text-faint mono mt-2">{res.details.tokens_generated} tokens</div>
            </div>
          )}
        </aside>
      </div>

      <div className="flex gap-2 mt-3">
        <input className="input" placeholder="Message…" value={input}
          onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()} />
        <button className="btn btn-primary" onClick={send} disabled={busy}><Send size={15} /></button>
      </div>
    </Page>
  );
}

function ArenaPanes({ rounds, busy, ready, leftId, rightId, setLeftId, setRightId }: {
  rounds: { prompt: string; left: string; right: string }[]; busy: boolean; ready: LoadedServer[];
  leftId: string; rightId: string; setLeftId: (v: string) => void; setRightId: (v: string) => void;
}) {
  const picker = (v: string, set: (s: string) => void) => (
    <select className="input" value={v} onChange={(e) => set(e.target.value)}>
      <option value="">mock backend</option>
      {ready.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
    </select>
  );
  return (
    <div className="grid grid-cols-2 gap-3 h-[30rem]">
      {[{ id: leftId, set: setLeftId, key: "left" as const }, { id: rightId, set: setRightId, key: "right" as const }].map((side) => (
        <div key={side.key} className="card p-4 flex flex-col anim-rise">
          {picker(side.id, side.set)}
          <div className="flex-1 overflow-auto space-y-3 pr-1 mt-3">
            {rounds.map((r, i) => (
              <div key={i} className="space-y-1">
                <div className="text-right"><span className="inline-block max-w-[85%] px-3 py-1.5 rounded-lg text-xs bg-primary text-primary-fg">{r.prompt}</span></div>
                <div className="inline-block max-w-[95%] px-3 py-2 rounded-lg text-sm bg-sunken">{r[side.key]}</div>
              </div>
            ))}
            {busy && <div className="text-faint text-sm pulse">denoising…</div>}
          </div>
        </div>
      ))}
    </div>
  );
}
