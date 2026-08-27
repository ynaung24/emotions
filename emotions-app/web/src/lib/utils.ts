import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function scoreColor(value: number): string {
  if (value >= 78) return "text-emerald-600 dark:text-emerald-400";
  if (value >= 60) return "text-brand-600 dark:text-brand-400";
  if (value >= 45) return "text-amber-600 dark:text-amber-400";
  return "text-rose-600 dark:text-rose-400";
}

export function scoreStroke(value: number): string {
  if (value >= 78) return "#10b981";
  if (value >= 60) return "#0080ff";
  if (value >= 45) return "#f59e0b";
  return "#f43f5e";
}
