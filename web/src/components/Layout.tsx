import { NavLink, Outlet, useLocation } from "react-router-dom";
import { LayoutDashboard, Boxes, Database, FlaskConical, Save, MessageSquare, Settings, FlaskRound } from "lucide-react";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/training", label: "Training", icon: FlaskConical },
  { to: "/export", label: "Save model", icon: Save },
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/settings", label: "Settings", icon: Settings },
];

export default function Layout() {
  const { pathname } = useLocation();
  return (
    <div className="min-h-screen flex">
      <aside className="w-60 shrink-0 border-r border-edge bg-surface flex flex-col">
        <div className="px-5 py-5 flex items-center gap-2.5">
          <span className="grid place-items-center w-8 h-8 rounded-lg bg-primary text-primary-fg">
            <FlaskRound size={17} />
          </span>
          <div className="leading-tight">
            <div className="text-sm font-semibold">dFactory Lab</div>
            <div className="text-[11px] text-faint">dLLM finetune</div>
          </div>
        </div>
        <nav className="flex-1 px-2.5 py-2 space-y-0.5">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}
              className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
              <n.icon size={17} strokeWidth={1.9} />
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="px-5 py-4 border-t border-edge text-[11px] text-faint">
          v0.1.0
        </div>
      </aside>

      <div className="flex-1 min-w-0 flex flex-col">
        <main className="flex-1 overflow-auto px-8 py-8">
          <div key={pathname} className="anim-rise max-w-[1180px] mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
