import { Component, ReactNode } from "react";
import ErrorFallback from "@/components/ErrorFallback";
import { isChunkLoadError } from "@/utils/chunkLoadError";

type ErrorBoundaryProps = {
  children: ReactNode;
  // the boundary clears itself when this changes, e.g. on navigation
  resetKey?: string;
};

type ErrorBoundaryState = {
  error: unknown;
};

/**
 * Without a boundary, React 19 unmounts the whole app on an uncaught render
 * error, including a lazy route chunk that failed to download, leaving a
 * blank page that only a manual reload recovers from.
 */
export default class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: unknown): ErrorBoundaryState {
    return { error };
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps) {
    if (this.state.error && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ error: null });
    }
  }

  render() {
    if (this.state.error) {
      return <ErrorFallback chunkError={isChunkLoadError(this.state.error)} />;
    }

    return this.props.children;
  }
}
