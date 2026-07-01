import { useApi } from "@/lib/hooks";

interface Security { bind: string; loopback: boolean; exposed: boolean; auth_required: boolean; token_present: boolean }
interface System { python: string; platform: string; cpus: number; hardware: { backend: string; device_count: number }; data_dir: string; engine_present: boolean }

export default function Settings() {
  const sec = useApi<Security>("/api/security", 0);
  const sys = useApi<System>("/api/system", 0);

  return (
    <div className="space-y-9">
      <header>
        <div className="eyebrow">Phase 9 · Security &amp; system</div>
        <h1 className="text-3xl mt-1">Settings</h1>
      </header>

      <section className="card p-5">
        <div className="eyebrow mb-3">Security posture</div>
        <div className="grid md:grid-cols-2 gap-x-8 gap-y-2 text-sm">
          <Row k="Bind" v={sec.data?.bind ?? "—"} />
          <Row k="Exposed" v={String(sec.data?.exposed ?? false)} warn={sec.data?.exposed} />
          <Row k="Auth required" v={String(sec.data?.auth_required ?? false)} ok={sec.data?.auth_required && sec.data?.exposed} />
          <Row k="Bootstrap token" v={sec.data?.token_present ? "set" : "auto (loopback)"} />
        </div>
        {sec.data?.exposed && (
          <p className="text-xs text-warn mt-3">
            Server is network-exposed. Requests must present the bootstrap token
            (<span className="mono">Authorization: Bearer &lt;token&gt;</span> or <span className="mono">?token=</span>).
          </p>
        )}
      </section>

      <section className="card p-5">
        <div className="eyebrow mb-3">System</div>
        <div className="grid md:grid-cols-2 gap-x-8 gap-y-2 text-sm">
          <Row k="Python" v={sys.data?.python ?? "—"} />
          <Row k="Platform" v={sys.data?.platform ?? "—"} />
          <Row k="CPUs" v={String(sys.data?.cpus ?? "—")} />
          <Row k="Hardware" v={sys.data?.hardware ? `${sys.data.hardware.backend} ×${sys.data.hardware.device_count}` : "—"} />
          <Row k="Data dir" v={sys.data?.data_dir ?? "—"} mono />
          <Row k="Engine vendored" v={sys.data?.engine_present ? "yes" : "no"} ok={sys.data?.engine_present} />
        </div>
      </section>

      <section className="card p-5">
        <div className="eyebrow mb-2">Design system — Editorial Paper (locked)</div>
        <p className="text-sm text-muted leading-relaxed">
          Warm paper base, ink text, a single terracotta accent, serif display + grotesque body + mono data.
          Deliberately distinct from — and not — the generic black+blue “AI-slop” SaaS look.
        </p>
        <div className="flex gap-2 mt-3">
          {["#faf7f2", "#1c1a17", "#bf431d", "#c9772a", "#e2dacb"].map((c) => (
            <span key={c} className="badge mono" style={{ background: c, color: c === "#faf7f2" || c === "#e2dacb" ? "#1c1a17" : "#fff7f0", borderColor: "transparent" }}>{c}</span>
          ))}
        </div>
      </section>
    </div>
  );
}

function Row({ k, v, mono, ok, warn }: { k: string; v: string; mono?: boolean; ok?: boolean; warn?: boolean }) {
  const cls = warn ? "text-warn" : ok ? "text-ok" : "";
  return (
    <div className="flex justify-between border-b rule py-1">
      <span className="eyebrow">{k}</span>
      <span className={`${mono ? "mono" : ""} ${cls} tnum`}>{v}</span>
    </div>
  );
}
