import { Component, lazy, type ReactNode, Suspense } from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { queryClient } from "@/lib/query";
import { Layout } from "@/components/Layout";
import { Spinner } from "@/components/ui/primitives";
import { Session } from "@/routes/Session";

// Results pulls in recharts (~400 KB) - keep it out of the initial bundle.
const Results = lazy(() =>
  import("@/routes/Results").then((m) => ({ default: m.Results })),
);

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="mx-auto max-w-md p-10 text-center">
          <h1 className="text-lg font-semibold">Something broke on this page.</h1>
          <p className="mt-2 text-sm text-muted">{this.state.error.message}</p>
          <button
            onClick={() => location.assign("/")}
            className="mt-4 rounded-lg bg-brand-500 px-4 py-2 text-sm text-white"
          >
            Back to start
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Layout>
            <Suspense
              fallback={
                <div className="grid place-items-center py-20">
                  <Spinner className="size-6" />
                </div>
              }
            >
              <Routes>
                <Route path="/" element={<Session />} />
                <Route path="/results" element={<Results />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Suspense>
          </Layout>
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}
