import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { Sparkles } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { WarmupBanner } from "./WarmupBanner";
import { cn } from "@/lib/utils";

export function Layout({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-10 border-b border-[var(--border)] surface/90 backdrop-blur">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3">
          <Link to="/" className="flex items-center gap-2 font-semibold">
            <Sparkles className="size-5 text-brand-500" />
            <span className="hidden sm:inline">Interview Evaluator</span>
            <span className="sm:hidden">Evaluator</span>
          </Link>
          <div className="flex items-center gap-1">
            <Link
              to="/"
              className={cn(
                "rounded-lg px-3 py-1.5 text-sm hover:surface-2",
                pathname === "/" && "text-brand-600 dark:text-brand-400 font-medium",
              )}
            >
              New session
            </Link>
            <ThemeToggle />
          </div>
        </div>
      </header>
      <WarmupBanner />
      <main className="mx-auto max-w-4xl px-4 py-6 sm:py-10">{children}</main>
    </div>
  );
}
