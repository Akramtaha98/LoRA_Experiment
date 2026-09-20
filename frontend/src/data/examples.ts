export type DemoExample = {
  name: string;
  description: string;
  context: string;
  question: string;
  reference: string;
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
  },
  {
    name: "Distractor",
    description: "Tests whether irrelevant facts distract the model.",
    context:
      "Mercury is the closest planet to the Sun. Venus is the second planet " +
      "from the Sun. Earth is the third planet from the Sun. Mars is known as the Red Planet.",
    question: "Which planet is second from the Sun?",
    reference: "Venus",
  },
  {
    name: "Hallucination Trap",
    description: "The requested fact does not exist in the supplied context.",
    context:
      "Dr. Sarah Williams joined the University of Bristol in 2018. " +
      "She specializes in computational linguistics and teaches natural language processing.",
    question: "Where did Dr. Sarah Williams earn her PhD?",
    reference: "The information is not provided in the context.",
  },
  {
    name: "Knowledge Conflict",
    description: "Tests whether the model follows context instead of pretrained knowledge.",
    context:
      "For this fictional experiment, the capital of France is Lyon.",
    question:
      "According to the provided context, what is the capital of France?",
    reference: "Lyon",
  },
];
