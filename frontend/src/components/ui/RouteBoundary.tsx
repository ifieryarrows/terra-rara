import { Component, type ErrorInfo, type ReactNode } from 'react';

type RouteBoundaryProps = { children: ReactNode; resetKey?: string };
type RouteBoundaryState = { failed: boolean; resetKey?: string };

export class RouteBoundary extends Component<RouteBoundaryProps, RouteBoundaryState> {
  state: RouteBoundaryState = { failed: false, resetKey: this.props.resetKey };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  // Reset when the route changes without remounting the shell. A `key` on the
  // boundary used to do this, but it also recreated AppShell on every
  // navigation and the header could start hidden from the outgoing scroll
  // position, flashing out and back in.
  static getDerivedStateFromProps(props: RouteBoundaryProps, state: RouteBoundaryState) {
    if (props.resetKey !== state.resetKey) return { failed: false, resetKey: props.resetKey };
    return null;
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Unable to render route', error, info.componentStack);
  }

  render() {
    if (this.state.failed) return <div className="cm-route-loading" role="alert"><h1>This view could not be loaded.</h1><p>Reload to retrieve the current version.</p><a href={window.location.href}>Reload this page</a><a href="/dashboard">Open dashboard</a></div>;
    return this.props.children;
  }
}
