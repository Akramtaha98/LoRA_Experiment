import assert from "node:assert/strict";
import fs from "node:fs";
import {
  matchesFilter,
  answerVerdict,
  explainAnswer,
  highlightParts,
  evidenceExcerpt,
  trainingChange,
  trainingPairs,
  trainingSummary,
  guidedExample,
  englishGuide,
  selectExamples,
  selectedExampleIndex,
} from "../src/lib/evidence.ts";
const examples = JSON.parse(
  fs.readFileSync(
    new URL("../public/data/compare-examples.json", import.meta.url),
  ),
);
const data = JSON.parse(
  fs.readFileSync(new URL("../public/data/results.json", import.meta.url)),
);
const fixture = {
  id: "fixture",
  question: "Gingrich",
  context: "Romney 46 peratus. Gingrich 32 peratus.",
  gold: "32 peratus",
  language: "malay",
  answers: {
    qwen3: { a: "2", em: 0, f1: 0 },
    mt5: { a: "19", em: 0, f1: 0 },
    t5gemma2: { a: "0", em: 1, f1: 1 },
  },
};
assert.equal(
  matchesFilter(fixture, "missed"),
  false,
  "a correct third model must exclude the all-missed filter",
);
assert.equal(matchesFilter(fixture, "disagree"), true);
assert.equal(matchesFilter(fixture, "t5gemma2"), true);
assert.equal(
  matchesFilter(
    {
      ...fixture,
      answers: { ...fixture.answers, t5gemma2: { a: "x", em: 0, f1: 0 } },
    },
    "missed",
  ),
  true,
);
assert.equal(
  matchesFilter(
    { ...fixture, answers: { qwen3: fixture.answers.qwen3 } },
    "missed",
  ),
  false,
);
assert.equal(
  answerVerdict({ a: "46 peratus", em: 0, f1: 0.5 }),
  "Some shared words",
);
assert.match(
  explainAnswer(fixture, { a: "46 peratus", em: 0, f1: 0.5 }),
  /Romney.*Gingrich/,
);
for (const e of examples) {
  for (const a of Object.values(e.answers)) {
    const parts = highlightParts(e.context, e.gold, a.a);
    assert.equal(
      parts.map((p) => p.text).join(""),
      e.context,
      "highlighting must preserve the entire original passage",
    );
    assert.ok(explainAnswer(e, a).length > 20);
  }
  const fail = matchesFilter(e, "missed");
  if (fail) assert.ok(Object.values(e.answers).every((a) => a.em !== 1));
}
assert.equal(
  highlightParts("<script>alert(1)</script>", "script")
    .map((p) => p.text)
    .join(""),
  "<script>alert(1)</script>",
);
assert.ok(
  highlightParts("abc abc", "abc").filter((p) => p.kind === "reference")
    .length === 2,
);
assert.ok(evidenceExcerpt(fixture.context, fixture.gold).includes("Gingrich"));
assert.equal(
  trainingChange(undefined, undefined).startsWith("A matched pair"),
  true,
);
assert.match(
  trainingChange(
    { generated: "a", em: 1, f1: 1 },
    { generated: "b", em: 0, f1: 0 },
  ),
  /mismatch/,
);
for (const language of ["arabic", "malay"]) {
  const e = guidedExample(examples, language);
  assert.ok(e);
  assert.ok(englishGuide(e));
}
let pairs = 0;
for (const item of data.compare.outputs)
  for (const model of Object.values(item.answers))
    for (const adapter of Object.values(model)) {
      const b = adapter.B_ce_lora,
        c = adapter.C_composite_lora;
      if (b && c) {
        pairs++;
        assert.ok(trainingChange(b, c));
      }
    }
assert.ok(pairs > 0, "training view needs real matched pairs");
const arabic = guidedExample(examples, "arabic");
assert.match(explainAnswer(arabic, arabic.answers.mt5), /feeding amount/);
const allExact = {
  ...fixture,
  answers: {
    qwen3: { a: "same", em: 1, f1: 1 },
    mt5: { a: "same", em: 1, f1: 1 },
    t5gemma2: { a: "same", em: 1, f1: 1 },
  },
};
assert.equal(matchesFilter(allExact, "matched"), true);
assert.equal(matchesFilter(allExact, "disagree"), false);
const allExamples = selectExamples(examples, "all");
assert.equal(
  examples.length,
  100,
  "the full requested 100-example set must remain available",
);
assert.ok(
  examples.every((e) =>
    ["qwen3", "mt5", "t5gemma2"].every((m) => e.answers[m]),
  ),
  "each saved example must show all three models",
);
assert.equal(allExamples.length, examples.length);
assert.equal(
  new Set(allExamples.map((e) => e.id)).size,
  examples.length,
  "every question must be selectable exactly once",
);
for (const language of ["arabic", "malay"]) {
  const filtered = selectExamples(examples, language);
  assert.equal(
    filtered.length,
    examples.filter((e) => e.language === language).length,
  );
  filtered.forEach((e) => assert.equal(e.language, language));
}
for (const [index, e] of allExamples.entries())
  assert.equal(selectedExampleIndex(allExamples, e.id), index);
assert.equal(selectedExampleIndex(allExamples, "no-longer-selected"), 0);
assert.deepEqual(
  selectExamples(examples, "all", "no-such-question-xyz-97654"),
  [],
);
assert.ok(selectExamples(examples, "all", "Gingrich").length > 0);
assert.ok(
  selectExamples(examples, "all", "", "missed").every((e) =>
    Object.values(e.answers).every((a) => a.em !== 1),
  ),
);
console.log(
  `evidence behavior checks passed (${examples.length} source examples, ${pairs} saved training pairs)`,
);

const expectedChanges = {
  qwen3: { qlora: 0, dora: 3 },
  mt5: { qlora: 5, dora: 7 },
  t5gemma2: { qlora: 4, dora: 4 },
};
for (const [model, adapters] of Object.entries(expectedChanges)) {
  for (const [adapter, changed] of Object.entries(adapters)) {
    assert.deepEqual(trainingSummary(data.compare.outputs, model, adapter), {
      total: 10,
      changed,
      unchanged: 10 - changed,
    });
    const filtered = trainingPairs(data.compare.outputs, model, adapter, true);
    assert.equal(filtered.length, changed);
    assert.ok(
      filtered.every(
        (item) =>
          item.answers[model][adapter].B_ce_lora.generated !==
          item.answers[model][adapter].C_composite_lora.generated,
      ),
    );
  }
}
assert.deepEqual(trainingSummary(data.compare.outputs, "missing", "qlora"), {
  total: 0,
  changed: 0,
  unchanged: 0,
});
assert.equal(
  trainingPairs(
    [{ answers: { qwen3: { qlora: { B_ce_lora: { generated: "a" } } } } }],
    "qwen3",
    "qlora",
  ).length,
  0,
);
