import { Outlet, NavLink } from 'react-router-dom';
import { Activity, LayoutDashboard, Brain, CheckCircle, Server } from 'lucide-react';
import clsx from 'clsx';
import { SpeedInsights } from '@vercel/speed-insights/react';

export const AppShell = () => {
  return (
    <div className="min-h-screen bg-[#020617] text-white selection:bg-copper-500/30">
      {/* Background gradients removed for professional UI per plan */}
      
      {/* Top Navigation */}
      <nav className="sticky top-0 z-50 bg-[#0f172a] border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3 shrink-0">
              <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center border border-slate-700">
                <Activity size={20} className="text-emerald-500" />
              </div>
              <div>
                <h1 className="text-lg font-bold tracking-tight text-white flex items-baseline gap-1.5">
                  Copper<span className="text-slate-400 font-light">Mind</span>
                </h1>
                <p className="text-[10px] text-slate-500 font-mono tracking-widest uppercase">
                  v2.0.0
                </p>
              </div>
            </div>

            {/* Main Nav Links */}
            <div className="hidden sm:flex items-center space-x-1">
              {[
                { to: "/", icon: LayoutDashboard, label: "Overview" },
                { to: "/models", icon: Brain, label: "Models" },
                { to: "/validation", icon: CheckCircle, label: "Validation" },
                { to: "/system", icon: Server, label: "System" },
              ].map(({ to, icon: Icon, label }) => (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) => clsx(
                    "flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                    isActive 
                      ? "bg-slate-800 text-white border border-slate-700" 
                      : "text-slate-400 hover:text-white hover:bg-slate-800/50"
                  )}
                >
                  <Icon size={16} />
                  {label}
                </NavLink>
              ))}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        <Outlet />
      </main>

      <SpeedInsights />
    </div>
  );
};
