import { Outlet, NavLink, Link } from 'react-router-dom';
import { LayoutDashboard, Brain, CheckCircle, Server } from 'lucide-react';
import { Brand } from '../components/ui/Brand';
const navigation = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Overview' },
  { to: '/models', icon: Brain, label: 'Models' },
  { to: '/validation', icon: CheckCircle, label: 'Validation' },
  { to: '/system', icon: Server, label: 'System' },
];
export function AppShell() {
  return <div className="cm-workspace">
    <a className="cm-skip" href="#main-content">Skip to workspace</a>
    <header className="cm-workspace-header"><div className="cm-workspace-bar"><Brand/><nav className="cm-workspace-nav" aria-label="Workspace">{navigation.map(({ to, icon: Icon, label }) => <NavLink to={to} key={to} end className="cm-nav-link"><Icon size={17} aria-hidden="true"/>{label}</NavLink>)}</nav></div></header>
    <main id="main-content" tabIndex={-1} className="cm-workspace-main"><Outlet/></main>
    <footer className="cm-workspace-footer"><span>CopperMind / Terra Rara · Research workspace</span><Link to="/">About the platform</Link></footer>
  </div>;
}
