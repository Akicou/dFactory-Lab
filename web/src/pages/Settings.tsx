import { Shield, Server, Palette } from "lucide-react";
import { useApi } from "../lib/hooks";
import { Page } from "../components/Page";

interface Security { bind: string; loopback: boolean; exposed: boolean; auth_required: boolean; token_present: boolean }
interface System { python: string; platform: string; cpus: number; hardware: { backend: string; device_count: number }; data_dir: string; engine_present: boolean }

export default function Settings() {
  const sec = useApi<Security>("/api/security", 0);
  const sys = useApi<System>("/api/system", 0);

  return (
    <Page title="Settings">
      <div className="grid md:grid-cols-2 gap-3 anim-stagger">
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-3"><Shield size={16} className="text-muted" /><span className="label">Security</span></div>
          <Row k="Bind" v={sec.data?.bind ?? "—"} />
          <Row k="Exposed" v={String(sec.data?.exposed ?? false)} warn={sec.data?.exposed} />
          <Row k="Auth required" v={String(sec.data?.auth_required ?? false)} ok={sec.data?.auth_required} />
          <Row k="Token" v={sec.data?.token_present ? "set" : "auto"} last />
        </div>

        <div className="card p-5">
          <div className="flex items-center gap-2 mb-3"><Server size={16} className="text-muted" /><span className="label">System</span></div>
          <Row k="Python" v={sys.data?.python ?? "—"} />
          <Row k="Hardware" v={sys.data?.hardware ? `${sys.data.hardware.backend} · ${sys.data.hardware.device_count}` : "—"} />
          <Row k="CPUs" v={String(sys.data?.cpus ?? "—")} />
          <Row k="Data dir" v={sys.data?.data_dir ?? "—"} mono last />
        </div>
      </div>

      <div className="card p-5">
        <div className="flex items-center gap-2 mb-3"><Palette size={16} className="text-muted" /><span className="label">Theme</span></div>
        <p className="text-sm text-muted">Clean white, soft corners, neutral palette, lucide icons.</p>
        <div className="flex gap-2 mt-3">
          {["#ffffff", "#fafafa", "#f4f4f5", "#18181b", "#71717a"].map((c) => (
            <span key={c} className="badge mono" style={{ background: c, color: c === "#ffffff" || c === "#fafafa" || c === "#f4f4f5" ? "#18181b" : "#fff", borderColor: "#ececef" }}>{c}</span>
          ))}
        </div>
      </div>
    </Page>
  );
}

function Row({ k, v, mono, ok, warn, last }: { k: string; v: string; mono?: boolean; ok?: boolean; warn?: boolean; last?: boolean }) {
  const cls = warn ? "text-warn" : ok ? "text-ok" : "";
  return (
    <div className={`flex justify-between py-2 text-sm ${last ? "" : "border-b border-edge"}`}>
      <span className="label">{k}</span>
      <span className={`${mono ? "mono" : ""} ${cls} tnum`}>{v}</span>
    </div>
  );
}
