export const MODELS = ["qwen3", "mt5", "t5gemma2"] as const;
export type Model = (typeof MODELS)[number];
export type Language = "arabic" | "malay";
export interface SavedAnswer {
  a: string;
  em: number;
  f1: number;
}
export interface SavedExample {
  id: string;
  config?: string;
  language: Language;
  question: string;
  context: string;
  gold: string;
  answers: Partial<Record<Model, SavedAnswer>>;
}
export type AnswerFilter = "all" | "matched" | "missed" | "disagree" | Model;

export function matchesFilter(
  example: SavedExample,
  filter: AnswerFilter,
): boolean {
  const answers = MODELS.map((m) => example.answers[m]);
  if (filter === "all") return true;
  if (MODELS.includes(filter as Model))
    return example.answers[filter as Model]?.em === 1;
  // All-model claims require every prediction; missing outputs cannot count as failures.
  if (!answers.every(Boolean)) return false;
  const exact = answers.filter((a) => a?.em === 1).length;
  if (filter === "matched") return exact === MODELS.length;
  if (filter === "missed") return exact === 0;
  return (
    new Set(
      answers.map((a) => a!.a.normalize("NFC").trim().replace(/\s+/g, " ")),
    ).size > 1
  );
}

export function answerVerdict(answer: SavedAnswer): string {
  return answer.em === 1
    ? "Exact reference match"
    : answer.f1 > 0
      ? "Some shared words"
      : "No reference match";
}

export function explainAnswer(
  example: SavedExample,
  answer: SavedAnswer,
): string {
  if (!answer.a.trim())
    return "The saved output is empty. The model did not provide an answer.";
  if (
    /Gingrich/i.test(example.question) &&
    /32/.test(example.gold) &&
    /46/.test(answer.a) &&
    !/32/.test(answer.a)
  )
    return "The answer selects Romney’s 46%, while the question asks about Gingrich’s 32%. Both numbers occur in the passage, but they refer to different people.";
  if (
    example.question === "ما هو اسم المجسات الصغيرة التي تحملها المشطيات؟" &&
    /عشرة\s+أضعاف/.test(answer.a)
  )
    return "The question asks for the name of small tentacles. This output starts with a feeding amount, then continues with species-count text. Those passage facts do not answer the question.";
  if (answer.em === 1)
    return "The output matches the reference after the experiment’s text normalization. This is an exact-match result, rather than a separate human judgment.";
  if (answer.a.length > Math.max(example.gold.length * 3, 70))
    return "The output is much longer than the reference. Compare its wording with the highlighted evidence to check whether it answers the question. Extra copied text can overlap the reference without answering it correctly.";
  if (example.context.includes(answer.a.trim()))
    return "This wording occurs in the passage, but it differs from the reference. Copying a passage span alone does not show that it answers the question.";
  if (answer.f1 > 0)
    return "The output shares words with the reference. Word overlap alone cannot establish that its meaning is correct; inspect the evidence and the differing words.";
  return "The output differs from the reference and has no token overlap under the recorded scoring rule. The saved scores alone do not explain why the model produced it.";
}

export interface HighlightPart {
  text: string;
  kind?: "reference" | "answer";
}
export function highlightParts(
  text: string,
  reference: string,
  answer = "",
): HighlightPart[] {
  const spans: { start: number; end: number; kind: "reference" | "answer" }[] =
    [];
  for (const [needle, kind] of [
    [reference, "reference"],
    [answer, "answer"],
  ] as const) {
    if (!needle.trim()) continue;
    let start = text.indexOf(needle);
    while (start >= 0) {
      spans.push({ start, end: start + needle.length, kind });
      start = text.indexOf(needle, start + needle.length);
    }
  }
  spans.sort((a, b) => a.start - b.start || (a.kind === "reference" ? -1 : 1));
  let cursor = 0;
  const parts: HighlightPart[] = [];
  for (const span of spans) {
    if (span.start < cursor) continue;
    if (span.start > cursor)
      parts.push({ text: text.slice(cursor, span.start) });
    parts.push({ text: text.slice(span.start, span.end), kind: span.kind });
    cursor = span.end;
  }
  if (cursor < text.length) parts.push({ text: text.slice(cursor) });
  return parts;
}

export function evidenceExcerpt(
  context: string,
  reference: string,
  answer = "",
): string {
  const sentences = context.match(/[^.!?؟\n]+[.!?؟]?/gu) ?? [context];
  const relevant = sentences.filter(
    (s) =>
      (reference.trim() && s.includes(reference)) ||
      (answer.trim() && s.includes(answer)),
  );
  return (relevant.length ? relevant : sentences.slice(0, 2)).join(" ").trim();
}

export function trainingChange(
  before: { generated: string; em: number; f1: number } | undefined,
  after: { generated: string; em: number; f1: number } | undefined,
): string {
  if (!before || !after)
    return "A matched pair is not available in the saved records. No answer or score has been filled in.";
  if (before.generated === after.generated)
    return "The saved answer is unchanged. The reward did not change the output in this example.";
  if (after.em > before.em)
    return "The output changed from a reference mismatch to an exact reference match.";
  if (after.em < before.em)
    return "The output changed from an exact reference match to a reference mismatch.";
  return "The wording changed. Compare the recorded exact-match and token-F1 scores below; this example alone does not establish a general improvement.";
}

export function guidedExample(
  examples: SavedExample[],
  language: Language,
): SavedExample | undefined {
  const inLanguage = examples.filter((e) => e.language === language);
  return (
    inLanguage.find((e) => /Gingrich/i.test(e.question) && /32/.test(e.gold)) ??
    inLanguage[0]
  );
}

export function englishGuide(
  example: SavedExample,
): { question: string; reference: string; evidence: string } | undefined {
  if (/Gingrich/i.test(example.question) && /32/.test(example.gold))
    return {
      question: "What percentage of the vote did Gingrich receive?",
      reference: "32 percent",
      evidence:
        "Romney won more than 46 percent of the vote. Gingrich came second with 32 percent.",
    };
  if (example.question === "ما هو اسم المجسات الصغيرة التي تحملها المشطيات؟")
    return {
      question: "What are the small tentacles carried by ctenophores called?",
      reference: "Tentilla (المويجسات)",
      evidence:
        "The passage describes the retractable tentacles as fringed with small tentacles called المويجسات.",
    };
  return undefined;
}
