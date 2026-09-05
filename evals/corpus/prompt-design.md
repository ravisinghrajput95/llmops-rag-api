# Prompt design for grounded answering

A grounding instruction tells the model to answer only from supplied context.
Without one, the model answers from parametric knowledge, which produces
fluent answers that the retrieved documents do not support.

A refusal instruction gives the model an explicit sentence to emit when the
context does not contain the answer. Specifying the exact wording matters,
because downstream code and evaluation both need to detect a refusal reliably.

Numbering the context passages and asking for citations in the form of bracketed
numbers makes an answer auditable. A reader can check each claim against the
passage it cites rather than trusting the answer wholesale.

Temperature controls sampling randomness. Low values suit extraction and
grounded question answering, where the desired behaviour is faithfulness to the
context rather than variety.
