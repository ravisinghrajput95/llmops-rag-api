# Language model pricing

Prices are quoted per million tokens and differ between input and output.
Output tokens are consistently more expensive, typically by a factor of four,
because generation is more computationally costly than reading.

The gpt-4o-mini model costs fifteen cents per million input tokens and sixty
cents per million output tokens. The larger gpt-4o costs two dollars fifty per
million input tokens and ten dollars per million output tokens, roughly
seventeen times more.

A token is about four characters of English on average, so a thousand-word
passage is roughly 1,300 tokens. Estimates based on word count understate cost
for text containing code or unusual vocabulary.

Retrieval-augmented generation is dominated by input tokens, because the
retrieved context is usually far longer than the question or the answer.
Reducing the number of retrieved chunks is therefore the most direct cost
lever available.
