import type { ModelScores } from "../lib/scoring";

export type DemoExample = {
  name: string;
  description: string;
  context: string;
  question: string;
  reference: string;
  /** Offline mock outputs so Compare works on Vercel without FastAPI. */
  mt5: ModelScores;
  qwen: ModelScores;
};

export const examples: DemoExample[] = [
  {
    name: "Simple Fact",
    description: "Basic answer extraction from supplied evidence.",
    context:
      "Marie Curie was a Polish and naturalized-French physicist and chemist. " +
      "She won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911.",
    question:
      "In what year did Marie Curie win the Nobel Prize in Chemistry?",
    reference: "1911",
    mt5: {
      answer: "1911",
      faithfulness: 0.94,
      exact_match: 1,
      f1: 1,
      latency_ms: 640,
    },
    qwen: {
      answer: "1911",
      faithfulness: 0.91,
      exact_match: 1,
      f1: 1,
      latency_ms: 410,
    },
  },
  {
    name: "Distractor",
    description: "Tests whether irrelevant facts distract the model.",
    context:
      "Mercury is the closest planet to the Sun. Venus is the second planet " +
      "from the Sun. Earth is the third planet from the Sun. Mars is known as the Red Planet.",
    question: "Which planet is second from the Sun?",
    reference: "Venus",
    mt5: {
      answer: "Venus",
      faithfulness: 0.88,
      exact_match: 1,
      f1: 1,
      latency_ms: 590,
    },
    qwen: {
      answer: "Mars",
      faithfulness: 0.22,
      exact_match: 0,
      f1: 0,
      latency_ms: 380,
    },
  },
  {
    name: "Hallucination Trap",
    description: "The requested fact does not exist in the supplied context.",
    context:
      "Dr. Sarah Williams joined the University of Bristol in 2018. " +
      "She specializes in computational linguistics and teaches natural language processing.",
    question: "Where did Dr. Sarah Williams earn her PhD?",
    reference: "The information is not provided in the context.",
    mt5: {
      answer: "The information is not provided in the context.",
      faithfulness: 0.86,
      exact_match: 1,
      f1: 1,
      latency_ms: 710,
    },
    qwen: {
      answer: "MIT",
      faithfulness: 0.08,
      exact_match: 0,
      f1: 0,
      latency_ms: 450,
    },
  },
  {
    name: "Knowledge Conflict",
    description: "Tests whether the model follows context instead of pretrained knowledge.",
    context: "For this fictional experiment, the capital of France is Lyon.",
    question:
      "According to the provided context, what is the capital of France?",
    reference: "Lyon",
    mt5: {
      answer: "Lyon",
      faithfulness: 0.97,
      exact_match: 1,
      f1: 1,
      latency_ms: 713,
    },
    qwen: {
      answer: "Paris",
      faithfulness: 0.31,
      exact_match: 0,
      f1: 0,
      latency_ms: 482,
    },
  },
];
