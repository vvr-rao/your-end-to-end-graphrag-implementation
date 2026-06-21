import ReactMarkdown from "react-markdown";

/** Render an answer string. For `deep_research` mode, split on the 7
 * section labels and render each as its own labelled block. For
 * `simple_qa` (or anything we can't parse), render as one paragraph. */

const DEEP_SECTIONS = [
  "SPECIFICS",
  "ANALYSIS",
  "ANSWER",
  "CONTRADICTIONS",
  "KEY CLAIMS",
  "COVERAGE IMBALANCE",
  "KEY INSIGHTS",
];

interface Section {
  label: string;
  body: string;
}

function splitDeepResearch(answer: string): Section[] | null {
  // Find each section label appearing on its own line.
  const lines = answer.split(/\r?\n/);
  const found: { idx: number; label: string }[] = [];
  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    if (DEEP_SECTIONS.includes(trimmed)) {
      found.push({ idx: i, label: trimmed });
    }
  }
  if (found.length < 3) return null; // not the structured form
  const out: Section[] = [];
  for (let i = 0; i < found.length; i++) {
    const start = found[i].idx + 1;
    const end = i + 1 < found.length ? found[i + 1].idx : lines.length;
    out.push({
      label: found[i].label,
      body: lines.slice(start, end).join("\n").trim(),
    });
  }
  return out;
}

export function AnswerView({
  answer,
  mode,
}: {
  answer: string | null;
  mode: string;
}) {
  if (!answer || !answer.trim()) {
    return (
      <p className="text-slate-400 italic">
        (no answer returned — see evidence below)
      </p>
    );
  }
  if (mode === "deep_research") {
    const sections = splitDeepResearch(answer);
    if (sections) {
      return (
        <div className="space-y-4">
          {sections.map((s) => (
            <section key={s.label}>
              <h3 className="text-xs font-semibold tracking-wider text-emerald-400 mb-1">
                {s.label}
              </h3>
              <div className="prose prose-invert prose-sm max-w-none text-slate-200">
                <ReactMarkdown>{s.body || "_None identified._"}</ReactMarkdown>
              </div>
            </section>
          ))}
        </div>
      );
    }
  }
  return (
    <div className="prose prose-invert prose-sm max-w-none text-slate-200">
      <ReactMarkdown>{answer}</ReactMarkdown>
    </div>
  );
}
