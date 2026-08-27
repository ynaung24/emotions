import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Check, ChevronDown, Search } from "lucide-react";
import { api } from "@/lib/api";
import type { Question } from "@/lib/types";
import { Badge, Card, CardBody, Spinner, Textarea } from "./ui/primitives";
import { cn } from "@/lib/utils";

interface Props {
  value: string;
  questionId: string | null;
  onChange: (text: string, id: string | null) => void;
}

export function QuestionPicker({ value, questionId, onChange }: Props) {
  const { data: questions, isLoading, isError, refetch } = useQuery({
    queryKey: ["questions"],
    queryFn: api.questions,
  });
  const [query, setQuery] = useState("");
  const [custom, setCustom] = useState(questionId === null && value.length > 0);

  const grouped = useMemo(() => {
    if (!questions) return [];
    const q = query.trim().toLowerCase();
    const matches = q
      ? questions.filter((x) => x.text.toLowerCase().includes(q) || x.category.includes(q))
      : questions;
    const by = new Map<string, Question[]>();
    for (const item of matches) {
      const list = by.get(item.category) ?? [];
      list.push(item);
      by.set(item.category, list);
    }
    return [...by.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [questions, query]);

  return (
    <Card>
      <CardBody className="space-y-3 pt-5">
        <div className="flex items-center justify-between">
          <label className="text-sm font-semibold uppercase tracking-wide text-muted">
            Interview question
          </label>
          <button
            type="button"
            onClick={() => {
              setCustom((c) => !c);
              onChange("", null);
            }}
            className="text-xs text-brand-600 hover:underline dark:text-brand-400"
          >
            {custom ? "Pick from list" : "Write my own"}
          </button>
        </div>

        {custom ? (
          <Textarea
            autoFocus
            placeholder="Type the interview question you want to practise…"
            value={value}
            onChange={(e) => onChange(e.target.value, null)}
            className="min-h-[4rem]"
          />
        ) : isLoading ? (
          <div className="flex items-center gap-2 py-6 text-sm text-muted">
            <Spinner /> Loading questions…
          </div>
        ) : isError ? (
          <div className="space-y-2 py-4 text-sm">
            <p className="text-rose-600 dark:text-rose-400">Couldn’t load the question list.</p>
            <button onClick={() => refetch()} className="text-brand-600 hover:underline">
              Retry
            </button>
          </div>
        ) : (
          <>
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-2.5 size-4 text-muted" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search questions or categories"
                className="w-full rounded-lg border border-[var(--border)] surface-2 py-2 pl-9 pr-3 text-sm focus-visible:outline-2 focus-visible:outline-brand-500"
              />
            </div>
            <div className="max-h-72 space-y-3 overflow-y-auto pr-1">
              {grouped.map(([category, items]) => (
                <div key={category}>
                  <div className="mb-1 flex items-center gap-2">
                    <ChevronDown className="size-3 text-muted" />
                    <span className="text-xs font-medium uppercase tracking-wide text-muted">
                      {category}
                    </span>
                  </div>
                  <ul className="space-y-1">
                    {items.map((q) => (
                      <li key={q.id}>
                        <button
                          type="button"
                          onClick={() => onChange(q.text, q.id)}
                          className={cn(
                            "flex w-full items-start gap-2 rounded-lg px-3 py-2 text-left text-sm hover:surface-2",
                            questionId === q.id && "surface-2 ring-1 ring-brand-400",
                          )}
                        >
                          <Check
                            className={cn(
                              "mt-0.5 size-4 shrink-0",
                              questionId === q.id ? "text-brand-500" : "opacity-0",
                            )}
                          />
                          {q.text}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
              {grouped.length === 0 && (
                <p className="py-4 text-sm text-muted">No questions match “{query}”.</p>
              )}
            </div>
          </>
        )}

        {value && (
          <div className="flex items-center gap-2 rounded-lg surface-2 px-3 py-2 text-sm">
            <Badge tone={questionId ? "brand" : "warn"}>
              {questionId ? "from corpus" : "custom question"}
            </Badge>
            <span className="line-clamp-2">{value}</span>
          </div>
        )}
      </CardBody>
    </Card>
  );
}
