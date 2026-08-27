import { forwardRef } from "react";
import type { ButtonHTMLAttributes, HTMLAttributes, TextareaHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "solid" | "outline" | "ghost";
  size?: "sm" | "md" | "lg";
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "solid", size = "md", ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500",
        "disabled:pointer-events-none disabled:opacity-50",
        size === "sm" && "h-8 px-3 text-sm",
        size === "md" && "h-10 px-4 text-sm",
        size === "lg" && "h-12 px-6 text-base",
        variant === "solid" && "bg-brand-500 text-white hover:bg-brand-600",
        variant === "outline" &&
          "border border-[var(--border)] surface hover:surface-2 text-[var(--text)]",
        variant === "ghost" && "hover:surface-2 text-[var(--text)]",
        className,
      )}
      {...props}
    />
  ),
);
Button.displayName = "Button";

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "surface rounded-xl border border-[var(--border)] shadow-sm",
        className,
      )}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-5 pt-5 pb-3", className)} {...props} />;
}

export function CardTitle({ className, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cn("text-sm font-semibold tracking-wide uppercase text-muted", className)} {...props} />;
}

export function CardBody({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-5 pb-5", className)} {...props} />;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        "w-full rounded-lg border border-[var(--border)] surface-2 px-3 py-2 text-sm leading-relaxed",
        "focus-visible:outline-2 focus-visible:outline-offset-0 focus-visible:outline-brand-500",
        "placeholder:text-muted resize-y min-h-[8rem]",
        className,
      )}
      {...props}
    />
  ),
);
Textarea.displayName = "Textarea";

export function Badge({
  className,
  tone = "neutral",
  ...props
}: HTMLAttributes<HTMLSpanElement> & { tone?: "neutral" | "brand" | "warn" }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
        tone === "neutral" && "surface-2 text-muted",
        tone === "brand" && "bg-brand-100 text-brand-700 dark:bg-brand-900 dark:text-brand-200",
        tone === "warn" && "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
        className,
      )}
      {...props}
    />
  );
}

export function Meter({ value, className }: { value: number; className?: string }) {
  return (
    <div className={cn("h-2 w-full overflow-hidden rounded-full surface-2", className)}>
      <div
        className="h-full rounded-full bg-brand-500 transition-[width] duration-500"
        style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
      />
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "size-5 animate-spin rounded-full border-2 border-[var(--border)] border-t-brand-500",
        className,
      )}
      role="status"
      aria-label="loading"
    />
  );
}
