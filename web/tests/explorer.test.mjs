import "./evidence.test.mjs";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  filterExperiments,
  compatible,
  matchedSamples,
  rankExperiments,
} from "../src/lib/experiments.ts";
const rows = JSON.parse(
  fs.readFileSync(new URL("../public/data/experiments.json", import.meta.url)),
).experiments;
assert.equal(
  filterExperiments(rows, {
    protocol: "original",
    model: "t5gemma2",
    config: "A",
  }).length,
  0,
  "missing configurations must not be represented as zero-score rows",
);
for (const r of rows)
  assert.ok(
    filterExperiments(rows, {
      protocol: r.protocol,
      model: r.model,
      config: r.config,
      adapter: r.adapter,
      language: r.language,
    }).some((e) => e.id === r.id),
  );
const a = rows.find((r) => r.id === "original:qwen3:B:qlora:arabic"),
  b = rows.find((r) => r.id === "original:qwen3:C:qlora:arabic");
assert.ok(compatible(a, b));
assert.equal(compatible(a, { ...b, protocol: "corrected" }), false);
assert.equal(compatible(a, { ...b, nRuns: 1 }), false);
assert.equal(compatible(a, { ...b, language: "malay" }), false);
assert.equal(compatible(a, { ...b, seeds: [7, 11, 13] }), false);
const sample = {
  seed: 42,
  index: 0,
  question: "Who?",
  context: "Ali arrived.",
  gold: "Ali",
  generated: "Ali",
  em: 1,
  f1: 1,
};
assert.equal(
  matchedSamples([sample], [{ ...sample, generated: "Other" }]).length,
  1,
);
assert.equal(matchedSamples([sample], [{ ...sample, seed: 123 }]).length, 0);
assert.equal(
  matchedSamples([sample], [{ ...sample, question: "When?" }]).length,
  0,
);
assert.equal(
  matchedSamples(
    [{ ...sample, question: null }],
    [{ ...sample, question: null }],
  ).length,
  0,
  "positions and gold strings do not establish question identity",
);
const selected = filterExperiments(rows, {
  protocol: "original",
  language: "arabic",
  config: "B",
});
const ranked = rankExperiments(selected, "em");
assert.ok(ranked.every((r, i) => i === 0 || ranked[i - 1].em >= r.em));
assert.equal(
  selected[0].id,
  rows.find(
    (r) =>
      r.protocol === "original" && r.language === "arabic" && r.config === "B",
  ).id,
  "ranking must not mutate underlying order",
);
console.log("explorer behavior checks passed");

assert.equal(
  compatible({ ...a, seeds: [], nEval: [] }, { ...b, seeds: [], nEval: [] }),
  false,
  "missing coverage cannot establish comparability",
);
assert.equal(
  matchedSamples([sample, sample], [sample]).length,
  0,
  "ambiguous duplicate items cannot be paired arbitrarily",
);
