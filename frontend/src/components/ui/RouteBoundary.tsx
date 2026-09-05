import { Component, type ErrorInfo, type ReactNode } from 'react';

export class RouteBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('Unable to render route', error, info.componentStack); }
  render() {
    if (this.state.failed) return <div className="cm-route-loading" role="alert"><h1>This view could not be loaded.</h1><p>Reload to retrieve the current version.</p><a href={window.location.href}>Reload this page</a><a href="/dashboard">Open dashboard</a></div>;
    return this.props.children;
  }
}
