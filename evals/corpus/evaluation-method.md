# Evaluating retrieval-augmented systems

Retrieval and generation fail differently and should be measured separately. A
wrong answer caused by retrieving the wrong document is a different defect from
a wrong answer generated despite correct context.

Retrieval hit rate measures whether the expected document appeared among the
retrieved chunks. The metric is only meaningful when the corpus is
substantially larger than the retrieval depth; retrieving four chunks from a
corpus of five makes a perfect score inevitable and uninformative.

Refusal accuracy measures whether out-of-corpus questions are declined. It is
the metric that detects hallucination, and it requires deliberately adversarial
cases, including questions on topics adjacent to the corpus.

A model-based judge scores answers using another language model. It captures
semantic correctness that keyword matching misses, at the cost of money per
evaluation run and non-determinism that makes it unsuitable for a build gate.
