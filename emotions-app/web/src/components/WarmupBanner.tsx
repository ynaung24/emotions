import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { api } from "@/lib/api";

// Polls /ready so the first evaluation doesn't look like a hang. The old app's
// global 10s axios timeout aborted the first real request and blamed the user's
// connection - this makes the cold start visible instead.
export function WarmupBanner() {
  const { data } = useQuery({
    queryKey: ["ready"],
    queryFn: api.ready,
    refetchInterval: (q) => (q.state.data?.ready ? false : 2_000),
    retry: false,
  });

  if (!data || data.ready) return null;

  return (
    <div className="border-b border-amber-300/50 bg-amber-100/70 dark:bg-amber-950/40">
      <div className="mx-auto flex max-w-4xl items-center gap-2 px-4 py-2 text-sm text-amber-800 dark:text-amber-300">
        <Loader2 className="size-4 animate-spin" />
        The scoring models are warming up. Your first evaluation may take a little longer.
      </div>
    </div>
  );
}
