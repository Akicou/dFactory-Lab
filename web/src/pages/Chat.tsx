import { useState } from "react";
import { MessageSquare, Send, Sparkles } from "lucide-react";
import { api } from "../lib/api";
import { Page } from "../components/Page";

interface GenResult { response: string; details: { diffusion_steps: number; unmasking: { step: number; unmasked_frac: number }[] | null; tokens_generated: number } }

export default function Chat() {
  const [msgs, setMsgs] = useState<{ role: string; content: string }[]>([{ role: "user", content: "Explain masked diffusion in one line." }]);
  const [input, setInput] = useState("");
  const [steps, setSteps] = useState(32);
  const [schedule, setSchedule] = useState("linear");
  const [showUnmask, setShowUnmask] = useState(true);
  const [res, setRes] = useState<GenResult | null>(null);
  const [busy, setBusy] = useState(false);

  const send = async () => {
    if (!input.trim()) return;
    const next = [...msgs, { role: "user", content: input }];
    setMsgs(next); setInput(""); setBusy(true);
    try {
      const r = await api<GenResult>("/api/chat/completions", {
        method: "POST",
        body: JSON.stringify({ messages: next, params: { diffusion_steps: steps, mask_schedule: schedule, show_unmasking: showUnmask } }),
      });
      setRes(r); setMsgs([...next, { role: "assistant", content: r.response }]);
    } catch { /* ignore */ } finally { setBusy(false); }
  };

  return (
    <Page title="Chat" icon={MessageSquare}>
      <div className="grid lg:grid-cols-[1fr_18rem] gap-3">
        <div className="card p-4 flex flex-col h-[30rem] anim-rise">
          <div className="flex-1 overflow-auto space-y-3 pr-1">
            {msgs.map((m, i) => (
              <div key={i} className={`anim-fade ${m.role === "user" ? "text-right" : ""}`}>
                <div className={`inline-block max-w-[85%] px-3.5 py-2 rounded-lg text-sm ${m.role === "user" ? "bg-primary text-primary-fg" : "bg-sunken"}`}>{m.content}</div>
              </div>
            ))}
            {busy && <div className="text-faint text-sm pulse">denoising…</div>}
          </div>
          <div className="flex gap-2 mt-3">
            <input className="input" placeholder="Message…" value={input}
              onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()} />
            <button className="btn btn-primary" onClick={send} disabled={busy}><Send size={15} /></button>
          </div>
        </div>

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

          {res?.details.unmasking && (
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
    </Page>
  );
}
