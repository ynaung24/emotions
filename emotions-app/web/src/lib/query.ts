import { QueryClient } from "@tanstack/react-query";

// ONE QueryClient for the whole app (the old app created two, nested, and the
// outer one's config was silently dead).
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60_000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});
