import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

export function Page({ title, icon: Icon, actions, children }: {
  title: string; icon?: LucideIcon; actions?: ReactNode; children: ReactNode;
}) {
  return (
    <div className="space-y-7">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          {Icon && <Icon size={20} strokeWidth={1.9} className="text-muted" />}
          <h1 className="text-2xl tracking-tight">{title}</h1>
        </div>
        {actions}
      </div>
      {children}
    </div>
  );
}

export function Card({ className = "", children }: { className?: string; children: ReactNode }) {
  return <div className={`card ${className}`}>{children}</div>;
}
