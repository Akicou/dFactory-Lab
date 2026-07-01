import { useState } from "react";
import { api } from "@/lib/api";

interface GenResult { response: string; details: { diffusion_steps: number; unmasking: { step: number; unmasked_frac: number }[] | null; tokens_generated: number } }

export default function Chat() {
  const [msgs, setMsgs] = useState<{ role: string; content: string }[]>([{ role: "user", content: "Explain block-diffusion in one line." }]);
  const [input, setInput] = useState("");
  const [modelDir, setModelDir] = useState("");
  const [steps, setSteps] = useState(32);
  const [schedule, setSchedule] = useState("linear");
  const [showUnmask, setShowUnmask] = useState(true);
  const [res, setRes] = useState<GenResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const send = async () => {
    if (!input.trim()) return;
    const next = [...msgs, { role: "user", content: input }];
    setMsgs(next); setInput(""); setBusy(true); setErr(null);
    try {
      const r = await api<GenResult>("/api/chat/completions", {
        method: "POST",
        body: JSON.stringify({
          model_dir: modelDir || undefined, messages: next,
          params: { diffusion_steps: steps, mask_schedule: schedule, show_unmasking: showUnmask },
        }),
      });
      setRes(r); setMsgs([...next, { role: "assistant", content: r.response }]);
    } catch (e) { setErr(String(e instanceof Error ? e.message : e)); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-7">
      <header>
        <div className="eyebrow">Phase 6 · Inference</div>
        <h1 className="text-3xl mt-1">Chat &amp; diffusion playground</h1>
        <p className="text-muted mt-2 max-w-2xl">Diffusion params first-class: denoising steps and masking schedule. Watch the per-step unmasking.</p>
      </header>

      <div className="grid lg:grid-cols-[1fr_18rem] gap-6">
        <section className="card p-4 flex flex-col h-[28rem]">
          <div className="flex-1 overflow-auto space-y-3 pr-1">
            {msgs.map((m, i) => (
              <div key={i} className={m.role === "user" ? "text-right" : ""}>
                <span className={`badge mono ${m.role === "user" ? "text-accent" : ""}`}>{m.role}</span>
                <div className={`mt-1 inline-block max-w-[85%] px-3 py-2 rounded text-sm ${m.role === "user" ? "bg-accent-soft" : "bg-surface-2"}`}>{m.content}</div>
              </div>
            ))}
            {busy && <div className="text-faint text-sm">denoising…</div>}
          </div>
          <div className="flex gap-2 mt-3">
            <input className="input" placeholder="message…" value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()} />
            <button className="btn btn-primary" onClick={send} disabled={busy}>send</button>
          </div>
          {err && <div className="text-danger mono text-xs mt-2">{err}</div>}
        </section>

        <aside className="space-y-4">
          <div className="card p-4 space-y-3">
            <div className="eyebrow">Diffusion params</div>
            <label className="block">
              <div className="flex justify-between text-xs"><span className="eyebrow">steps</span><span className="mono tnum">{steps}</span></div>
              <input type="range" min={1} max={256} value={steps} onChange={(e) => setSteps(Number(e.target.value))} className="w-full accent-[var(--color-accent)]" />
            </label>
            <label className="block">
              <span className="eyebrow">mask schedule</span>
              <select className="input mt-1" value={schedule} onChange={(e) => setSchedule(e.target.value)}>
                <option>linear</option><option>geometric</option><option>cosine</option>
              </select>
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={showUnmask} onChange={(e) => setShowUnmask(e.target.checked)} /> show unmasking
            </label>
            <label className="block">
              <span className="eyebrow">model dir</span>
              <input className="input mt-1 mono text-xs" value={modelDir} onChange={(e) => setModelDir(e.target.value)} placeholder="(mock if empty)" />
            </label>
          </div>

          {res?.details.unmasking && (
            <div className="card p-4">
              <div className="eyebrow mb-2">Unmasking · {res.details.diffusion_steps} steps</div>
              <div className="flex items-end gap-px h-20">
                {res.details.unmasking.map((s) => (
                  <div key={s.step} className="flex-1 bg-accent rounded-t" style={{ height: `${s.unmasked_frac * 100}%` }} title={`step ${s.step + 1}: ${Math.round(s.unmasked_frac * 100)}%`} />
                ))}
              </div>
              <div className="text-xs text-faint mono mt-2">{res.details.tokens_generated} tokens</div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
