import { NavLink, Outlet } from "react-router-dom";

const NAV = [
  { to: "/", label: "Dashboard", end: true, glyph: "◷" },
  { to: "/models", label: "Models & Merge", glyph: "▤" },
  { to: "/datasets", label: "Datasets", glyph: "≣" },
  { to: "/training", label: "Training", glyph: "△" },
  { to: "/export", label: "Export", glyph: "↗" },
  { to: "/chat", label: "Chat", glyph: "✶" },
  { to: "/settings", label: "Settings", glyph: "⚙" },
];

export default function Layout() {
  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-60 shrink-0 border-r rule bg-surface-2/60 flex flex-col">
        <div className="px-5 py-5 border-b rule">
          <div className="eyebrow">dFactory · Lab</div>
          <div className="font-serif text-xl leading-tight mt-0.5">
            dLLM <span className="text-accent">finetune</span>
          </div>
        </div>
        <nav className="flex-1 px-2 py-3 space-y-0.5">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
            >
              <span className="text-faint w-4 text-center">{n.glyph}</span>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="px-5 py-4 border-t rule">
          <div className="eyebrow mb-1">Editorial Paper</div>
          <p className="text-xs text-muted leading-relaxed">
            Warm paper · terracotta accent · serif headlines. Not the black+blue SaaS look.
          </p>
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="h-14 border-b rule flex items-center justify-between px-8 bg-paper/80 backdrop-blur">
          <div className="eyebrow">Diffusion LLM Laboratory</div>
          <a className="text-xs text-muted hover:text-accent" href="/api/docs" target="_blank" rel="noreferrer">
            API reference ↗
          </a>
        </header>
        <main className="flex-1 overflow-auto px-8 py-7 max-w-[1200px] w-full">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
