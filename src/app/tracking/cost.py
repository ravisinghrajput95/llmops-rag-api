"""Token -> money estimation.

Prices are USD per 1M tokens, as published by OpenAI. They are baked in
deliberately: this project's entire premise is knowing what a call costs
*before* the invoice arrives. Update PRICING when OpenAI changes list prices.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float = 0.0


# USD per 1,000,000 tokens.
PRICING: dict[str, ModelPrice] = {
    "gpt-4o-mini": ModelPrice(input_per_1m=0.15, output_per_1m=0.60),
    "gpt-4o": ModelPrice(input_per_1m=2.50, output_per_1m=10.00),
    "gpt-4.1-mini": ModelPrice(input_per_1m=0.40, output_per_1m=1.60),
    "text-embedding-3-small": ModelPrice(input_per_1m=0.02),
    "text-embedding-3-large": ModelPrice(input_per_1m=0.13),
}

# Used when the configured model isn't in the table. gpt-4o-mini pricing is a
# reasonable floor; an unknown model is more likely to cost more, not less, so
# treating the estimate as approximate is safer than reporting zero.
_FALLBACK = PRICING["gpt-4o-mini"]


def price_for(model: str) -> ModelPrice:
    if model in PRICING:
        return PRICING[model]
    # Tolerate dated snapshots like "gpt-4o-mini-2024-07-18".
    for known, price in PRICING.items():
        if model.startswith(known):
            return price
    return _FALLBACK


def estimate_cost_usd(model: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> float:
    price = price_for(model)
    usd = (
        prompt_tokens * price.input_per_1m + completion_tokens * price.output_per_1m
    ) / 1_000_000
    # Sub-cent precision matters here; a query costs ~$0.0002.
    return round(usd, 8)


def usd_to_inr(usd: float, rate: float = 88.0) -> float:
    return round(usd * rate, 6)
