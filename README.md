# LLMOps RAG API — Cloud Run, MLflow, and a hard budget ceiling

[![CI](https://github.com/ravisinghrajput95/llmops-rag-api/actions/workflows/ci.yml/badge.svg)](https://github.com/ravisinghrajput95/llmops-rag-api/actions/workflows/ci.yml)
[![Deploy](https://github.com/ravisinghrajput95/llmops-rag-api/actions/workflows/deploy.yml/badge.svg)](https://github.com/ravisinghrajput95/llmops-rag-api/actions/workflows/deploy.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-185%20passing-brightgreen)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A production-shaped RAG service built to run on **₹0 of GCP spend**: FastAPI +
Chroma + OpenAI, deployed to Cloud Run with keyless GitHub Actions CI/CD, with
every request's latency, token usage and estimated cost logged to MLflow.

What makes it LLMOps rather than a RAG demo is everything downstream of the
answer: a 128-case golden set with a quality gate in CI, prompts versioned
behind a content-addressed lock so a wording change cannot slip into eval
history unnoticed, retrieval parameters chosen by measured sweep rather than by
guess, and drift detection that watches live traffic which has no labels at all.

| Measured | |
|---|---|
| Eval (128 cases, `gpt-4o-mini`) | 98.4% accuracy · 100% retrieval · 98.5% refusal · 100% citations |
| Cost per query | ~₹0.03, measured not modelled |
| Cost per full eval run | $0.0094 |
| GCP spend at demo scale | ₹0 — every resource inside Always Free |

Built under a specific constraint — **₹266 of GCP trial credit expiring
15 Sep 2026** — so cost is treated as a first-class design input, not a
footnote. Every architectural decision below is justified in terms of what it
costs.

---

## Read this first: the two bills

These are separate, and conflating them is the fastest way to get surprised.

| | Pays for | Funded by | Runs out |
|---|---|---|---|
| **Google Cloud** | Cloud Run, storage, registry | GCP trial credit, then Always Free quotas | Credit expires 15 Sep 2026; the service keeps running on Always Free |
| **OpenAI** | `gpt-4o-mini` + embedding calls | Your OpenAI account balance | Independent of GCP entirely |

**GCP trial credits do not pay for OpenAI.** The design keeps GCP usage inside
Always Free quotas, so the only thing you actually spend on is OpenAI tokens —
roughly **₹0.03 per query** (measured, see [Cost model](#cost-model)).

---

## Architecture

```
                       ┌──────────────────────────────┐
   POST /ingest ──────▶│  FastAPI (Cloud Run)         │
   POST /query  ──────▶│  min-instances=0             │
   GET  /health ──────▶│  max-instances=2             │
                       └───────┬──────────────┬───────┘
                               │              │
                    embed +    │              │  structured JSON logs
                    retrieve   │              └──────────▶ Cloud Logging
                               ▼                            (50 GiB/mo free)
                       ┌───────────────┐
                       │ Chroma        │  embedded, /tmp, no server
                       └───────┬───────┘
                               │ top-k chunks
                               ▼
                       ┌───────────────┐
                       │ OpenAI        │  gpt-4o-mini + text-embedding-3-small
                       │ gpt-4o-mini   │  ← the only thing that costs money
                       └───────┬───────┘
                               │ latency, tokens, cost
                               ▼
                       ┌───────────────┐
                       │ MLflow        │  SQLite backend + GCS artifacts
                       └───────────────┘
```

### Why these choices

| Decision | Alternative rejected | Reason |
|---|---|---|
| Cloud Run | GCE VM / GKE | A VM bills 24/7 whether or not anyone calls it. Cloud Run at `min-instances=0` bills only per request, and 2M requests/month are free forever. |
| OpenAI API | Self-hosted Llama on a GPU VM | The cheapest GCP GPU is roughly ₹400+/day. That alone would exhaust the ₹266 credits before the first day ended. |
| OpenAI embeddings | Local `sentence-transformers` | Would pull `torch` into the image (~2 GB), blowing past Artifact Registry's free 0.5 GB and adding 30s+ to cold starts. Embeddings cost $0.02/1M tokens — effectively free. |
| Chroma (embedded) | Pinecone / Vertex Vector Search | Managed vector DBs have no meaningful free tier. Chroma runs in-process. |
| `mlflow-skinny` | Full `mlflow` | Full MLflow drags in pandas, scipy, alembic tooling and Docker helpers. Skinny keeps the same tracking API at a fraction of the image size. |
| SQLite + GCS | Cloud SQL | Cloud SQL has **no free tier** — roughly ₹700/month minimum. That is 2.6× the entire remaining budget. |
| Workload Identity Federation | JSON service account key | A long-lived key in GitHub secrets never expires; if it leaks, someone else spends your credits. WIF tokens live for minutes. Both are free. |

---

## Cost model

Measured from the cost table in `src/app/tracking/cost.py`, which is unit-tested
against OpenAI's published rate card.

**Per query** (~1,200 prompt tokens + 250 completion tokens on `gpt-4o-mini`):

```
input   1,200 tokens × $0.15/1M  = $0.00018
output    250 tokens × $0.60/1M  = $0.00015
embed      20 tokens × $0.02/1M  = $0.0000004
                                   ─────────
                          total  ≈ $0.00033  ≈  ₹0.029
```

**≈ 34 queries per rupee** at those token counts. That worked example assumes
a fairly full 1,200-token context; the numbers actually measured on this
corpus are lower, because a small corpus retrieves a smaller prompt.

### Measured, not modelled

From a real `make eval` run against `gpt-4o-mini` (15 questions, 5 chunks
indexed) on 2026-09-05:

| | Measured |
|---|---|
| Cost, 15 queries | **$0.001310** |
| Cost per query | **$0.0000873** ≈ ₹0.0077 |
| Queries per rupee | **≈ 130** |
| Mean latency | 1,402 ms |
| p95 latency | 1,610 ms |

So the modelled `$0.00033` is conservative by ~3.8x against this corpus —
which is the right direction for an estimate to be wrong in, and worth knowing
before trusting the table above on a larger corpus.

Your ₹266, if it were spendable on OpenAI (it is not — it is GCP credit),
would be tens of thousands of queries at this rate. In practice GCP spend
stays at zero and OpenAI is billed separately.

The API returns this on every call, so it is observable rather than theoretical:

```json
{
  "estimated_cost_usd": 0.00033,
  "estimated_cost_inr": 0.029,
  "usage": {"prompt_tokens": 1200, "completion_tokens": 250, "embedding_tokens": 20},
  "latency_ms": 812.4,
  "retrieval_ms": 41.2,
  "generation_ms": 770.1
}
```

### Cost controls built into the code

- **Similarity floor + short-circuit** — a vector search over a non-empty
  collection always returns *something*, so an unrelated question would still
  retrieve a junk chunk and pay for a completion. `MIN_SIMILARITY` (default
  `0.2`) drops chunks below the threshold; when nothing survives, `/query`
  returns "I don't know" *without calling the LLM at all*. Measured on the live
  service: a relevant chunk scored **0.62**, a loosely related one **0.15**, an
  unrelated question **0.10**. The threshold is embedding-model specific —
  recalibrate if you change models, and lower it if legitimate questions start
  returning "I don't know". Enforced by a test.
- **`max_instances = 2`** — a hard ceiling. If the public endpoint gets
  scraped, the blast radius is bounded.
- **`--timeout 60s`** — a hung upstream call cannot bill for minutes.
- **`cpu_idle = true`** — CPU is billed only while a request is in flight.
- **Artifact Registry cleanup policy** — keeps 3 recent images, deletes
  untagged ones after 7 days, so CI does not silently grow past the free 0.5 GB.
- **GCS 30-day lifecycle rule** — MLflow artifacts self-delete before they
  approach the free 5 GB.
- **`min_instances` is validated to be 0** in Terraform — changing it requires
  deliberately editing a validation block.

---

## GCP resources: Always Free vs. trial credits

> The whole point of this table: **nothing here should touch your ₹266** at
> demo scale. The items marked ⚠️ are the ones that *could* if you let them grow.

### Always Free — no expiry, survives your trial ending

| Resource | Free allowance | This project's usage |
|---|---|---|
| **Cloud Run** | 2M requests, 360k GiB-s, 180k vCPU-s per month | A demo uses a rounding error of this. **Only true while `min-instances=0`.** |
| **Cloud Logging** | 50 GiB ingest per project per month | Structured JSON logs; nowhere near it. |
| **Secret Manager** | 6 active secret versions, 10k access ops/month | 2 secrets. |
| **Service accounts / IAM / WIF** | Unlimited | Free by definition. |
| **Artifact Registry** | 0.5 GB storage per month | ⚠️ **~162 MB compressed per image** (measured). Cleanup policy keeps 3. Layers are shared between versions, so 3 builds of the same dependency set store ~162 MB total, not 3×. **Without it, CI crosses this in a couple of weeks.** |
| **Cloud Storage** | 5 GB-months **in us-central1, us-east1, us-west1 only** | ⚠️ MLflow artifacts. 30-day lifecycle rule. **Any other region bills from byte one** — Terraform validates the region for this reason. |

### Draws down trial credits ⚠️

| Resource | When it starts costing | Guard in place |
|---|---|---|
| Cloud Run with `min-instances > 0` | Immediately, 24/7, even with zero traffic | Terraform `validation` block rejects it |
| Artifact Registry over 0.5 GB | ~2 images beyond the cleanup policy | `cleanup_policies` keeps 3 |
| Cloud Storage over 5 GB, or outside US regions | Immediately if the region is wrong | Region `validation` + lifecycle rule |
| Egress beyond 1 GB/month to the internet | High-traffic responses | `max_instances = 2` bounds it |

### Never provisioned — and why

Compute Engine VMs, GKE, Cloud SQL, Memorystore, Vertex AI endpoints, load
balancers, static IPs, NAT gateways. Each of these bills hourly regardless of
traffic. `scripts/cost_check.sh` actively scans for them and flags any that
appear.

---

## Quick start (local)

```bash
git clone <your-repo-url> && cd llmops-rag-api

make venv install          # Python 3.12 venv, matching the container
cp .env.example .env       # then put your OPENAI_API_KEY in it

make test                  # 68 tests, fully mocked — no network, no spend
make run                   # http://localhost:8080/docs
```

Try it end to end:

```bash
./scripts/smoke_test.sh    # health → ingest → grounded query → ungrounded query
```

Or by hand:

```bash
curl -X POST localhost:8080/ingest -H 'Content-Type: application/json' -d '{
  "documents": [{"text": "Cloud Run scales to zero when idle, so it costs nothing."}]
}'

curl -X POST localhost:8080/query -H 'Content-Type: application/json' -d '{
  "question": "What happens when Cloud Run is idle?"
}' | jq
```

Inspect the tracked runs:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///./data/mlflow.db
# http://localhost:5000 — one run per request, with latency/tokens/cost
```

---

## Deployment

### 1. Provision (once)

**Terraform** (recommended — it encodes the cost guardrails):

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars   # set project_id, github_repo
export TF_VAR_openai_api_key="sk-..."
export TF_VAR_app_api_key="$(openssl rand -hex 24)"   # protects your endpoints

terraform init
terraform plan       # read this before applying
terraform apply
terraform output github_secrets_summary
```

**Or plain gcloud**, if you prefer to see the API calls:

```bash
export PROJECT_ID=my-project GITHUB_REPO=owner/repo OPENAI_API_KEY=sk-...
./scripts/bootstrap.sh
./scripts/setup_wif.sh
```

### 2. Set a budget alert — do this on day one

```bash
PROJECT_ID=my-project BUDGET_INR=200 ./scripts/set_budget.sh
```

A budget **alerts, it does not cap**. Google will happily keep billing past it.
It exists to turn a silent drain into an email you can act on.

### 3. Wire up CI/CD

Add the repository secrets printed by `terraform output github_secrets_summary`
(or `setup_wif.sh`), then:

```bash
git push origin main
```

`.github/workflows/deploy.yml` then runs: lint → tests → build → push to
Artifact Registry → deploy to Cloud Run → smoke-test `/health` → **roll back
automatically if the smoke test fails**.

Authentication is keyless via Workload Identity Federation, scoped by an
`attribute_condition` to your repository alone.

---

## API

| Method | Path | Auth | Cost |
|---|---|---|---|
| `GET` | `/health` | none | free — does no I/O |
| `GET` | `/ready` | none | free — reports collection size |
| `POST` | `/ingest` | `X-API-Key` | embedding tokens only |
| `POST` | `/ingest/file` | `X-API-Key` | embedding tokens only |
| `POST` | `/query` | `X-API-Key` | embeddings + completion |
| `GET` | `/docs` | none | free — interactive OpenAPI UI |

Auth is enabled only when `APP_API_KEY` is set; unset, local development stays
frictionless. `/health` deliberately stays open so uptime probes work.

---

## Durability: how state survives scale-to-zero

Cloud Run's filesystem is read-only except `/tmp`, which is an **in-memory
tmpfs private to one instance**. Left alone, that means documents ingested by
one instance are invisible to another and everything vanishes a few idle
minutes later.

So the Chroma directory is snapshotted to Cloud Storage after any ingest that
changed something, and restored when a new instance starts. The restore runs
before Chroma opens the directory, because `PersistentClient` reads its SQLite
file and HNSW index at construction.

```
POST /ingest ──► embed ──► Chroma (/tmp) ──► tar.gz ──► gs://<bucket>/snapshots/
cold start   ──► restore from GCS ──► Chroma opens ──► ready
POST /query  ──► Chroma (/tmp)                    # never touches GCS
```

Set `GCS_BUCKET` to enable it; leave it empty locally, where the filesystem is
already durable. Terraform wires it to the same Always-Free bucket that holds
MLflow artifacts, so durability costs **nothing**.

What this deliberately does not solve:

| Limitation | Why it is acceptable here |
|---|---|
| Last write wins under concurrent ingest | Needs GCS generation preconditions and a retry loop. Ingest is rare and `max-instances` is 2. The restored generation is logged, so a lost write is diagnosable. |
| Whole-directory snapshots | Chroma's SQLite file and HNSW index must move together or the collection is corrupt. |
| MLflow's SQLite run history is still ephemeral | Artifacts persist (they go to GCS). For durable run history, point `MLFLOW_TRACKING_URI` at a real backend. |

A GCS outage degrades this to the old ephemeral behaviour rather than taking
the service down: snapshot failures are logged and swallowed.

The alternative was Cloud SQL with pgvector at ~₹700/month — **2.6x the entire
remaining budget** this project was built against.

---

## Evaluation

Cost and latency were always measured; answer quality was not. A 128-case
golden set (`evals/golden.jsonl`) over a 22-document corpus (`evals/corpus/`)
now scores:

| Metric | What it catches |
|---|---|
| `retrieval_hit_rate` | The expected document was retrieved. Chunking and similarity-floor regressions show up here first. |
| `keyword_hit` | The answer contains a fact the corpus supports. |
| `refusal_accuracy` | Out-of-corpus questions get "I don't know". **Floor is 98%** — a confident hallucination is worse than no answer, and the floor admits exactly one known failure (see below). |
| `citation_rate` | Answers cite passages as `[n]`, as the prompt requires. |

**67 of the 128 cases are out-of-corpus**, and 34 of those are deliberate
near-misses — questions on topics the corpus covers whose specific fact it
lacks ("what is Cloud Run's maximum request timeout?"). That tier exists
because an earlier refusal metric measured over eight mostly-easy negatives
(weather, recipes, AWS) was flattering the system by 26 points; see
[Why the floor stops here](#why-the-floor-stops-here). The rest are adjacent
technologies, questions about the collection itself, plainly unrelated
subjects, and instruction-override attempts. The grounded cases deliberately
include numeric precision, negation, conditional consequences and
cross-document questions.

**Latest run** (`make eval`, `gpt-4o-mini`, prompt v2, 2026-09-06 — 128 cases, 22 documents):

| Metric | Result |
|---|---|
| accuracy | **98.4%** (126/128) |
| retrieval hit rate | **100%** (61 grounded cases) |
| refusal accuracy | 98.5% (66/67) |
| citation rate | **100%** |
| mean / p95 latency | 1,098 ms / 1,505 ms |
| total cost | $0.009432 |

Gate **passed**. The two remaining failures are `gpt4o-vs-mini` (the model
quoted both prices instead of the ratio the corpus states; intermittent) and
`near-cr-maxconc`, described under "Prompt versions that were measured" below.

This is not comparable to the previous 100%, because the set got much harder
in between: 60 negatives were added, 34 of them near-misses on topics the
corpus covers. Cost per case still fell 30% ($0.000100 → $0.000070), and no
failure at any point was a retrieval failure — the hit rate has held at 100%
throughout.

**Run-to-run variance is about two cases** at `temperature=0.2`, which is worth
knowing before reading anything into a one-case difference. `near-cr-maxconc`
is the clearest example: the shipped prompt refuses it correctly in roughly one
run out of five and answers it wrongly in the rest.

**Why this 100% means something and the previous one did not.** The corpus was
three documents — five chunks — against `top_k=4`. Every query retrieved ~80%
of the entire corpus, so a perfect retrieval score was arithmetically
inevitable and measured nothing. The corpus is now 22 documents / 42 chunks,
and a query retrieves **10%** of it. Scoring 100% under that selection
pressure is evidence; scoring it over five chunks was not.

The documents are deliberately near-neighbours — `cloud-run` (cost) beside
`cloud-run-scaling` (concurrency), `storage` beside `gcs-lifecycle` and
`artifact-registry`, `mlflow` beside `openai-pricing` — so a case like
"what is the cost effect of setting concurrency to one?" fails unless
retrieval discriminates between two documents that both discuss cost.

`tests/test_evaluation.py` enforces this property directly: a test fails if
the corpus ever shrinks back toward the retrieval depth.

### Prompt versions that were measured

Versioning a prompt makes a change *visible* in eval history. `make compare-prompts`
makes one *decidable*: both arms run against the same store, corpus and retrieval
config, so the only variable is the wording. Every candidate below was scored
over the full 128-case set.

| variant | system prompt | outcome |
|---|---|---|
| `v2-precision` | 951 chars | **Rejected.** Told the model to answer the supported part and name the unsupported part. It named it *using the refusal sentence*, so a partially-correct answer scored as a refusal — the exact opposite of the intent. |
| `v3-precision` | 1029 chars | **Rejected on cost.** Fixed that backfire and reached 98.4% accuracy, but **+30%** per query for an accuracy difference inside the noise band. Most of its length was the rules that did not work. |
| `v4-minimal` → **v2** | 439 chars | **Shipped.** Keeps only the rule that repeatedly worked — do simple comparison or arithmetic rather than declining — for **+5.7%**. Fixed `archive-minimum-duration` in every run, and took citation rate to 100%. |

The honest reading is that the aggregate accuracy gain is inside the noise. The
justification for shipping v2 is the specific reproducible fix and the citation
rate, not the headline number.

**What no prompt fixed** is `near-cr-maxconc`. Asked for Cloud Run's maximum
concurrency, the model reports the documented *default* of eighty as a maximum.
An explicit "a default is not a limit" instruction did not help, and it fails
intermittently rather than always. It is a reading error on the one passage that
should have been retrieved — the same wall the similarity floor hits, one layer
up. `Thresholds.refusal_accuracy` sits at 0.98 to admit exactly this and trip on
a second.

### Production monitoring

The eval measures accuracy against a golden set. Production cannot: nobody
labels a real question, so there is no accuracy number to watch. `make drift`
compares recent traffic against the conditions quality was last measured under.
It reads MLflow, calls no model, and spends nothing.

**The signal that makes it possible.** `refused` is now recorded on every
query, and the eval established what a refusal means: the model refused 66 of
67 out-of-corpus questions and only 2 of 61 answerable ones. A refusal is
therefore a reasonably calibrated statement that the corpus could not answer
that question, and the refusal rate over a window estimates how much traffic
the corpus cannot serve — a quality measurement extracted from an unlabelled
stream.

**Where the data comes from.** Cloud Run's filesystem is a per-instance tmpfs,
so the MLflow tracking database — the params, metrics and tags this reads — died
on every scale-to-zero while only artifacts reached GCS. Drift detection was
blind to production by construction. It is now snapshotted into the same
Always-Free bucket as the Chroma store, every `MLFLOW_SNAPSHOT_EVERY` runs (25)
plus once on shutdown. Batching is deliberate: Cloud Storage's free tier allows
5,000 class A operations a month and a write per query would spend them.

**One object per instance, not one shared object.** The Chroma snapshot can use
last-write-wins because ingest is rare. This is written every few dozen queries
by every instance, so at `max-instances=2` a shared object would have instances
replacing each other's runs wholesale — losing roughly half of them and keeping
whichever wrote last. Each process instead writes `snapshots/mlflow/<revision>-<id>.tar.gz`
and never reads anyone else's; `make drift` merges the shards when it reads.
Sharding removes the conflict rather than arbitrating it, which is why there is
no locking here. An instance therefore never restores this file — it owns its
shard and starts empty, which is exactly what its shard should contain.

Two bounded costs, both stated rather than hidden: up to `MLFLOW_SNAPSHOT_EVERY`
runs are lost if an instance dies between snapshots, and the bucket's lifecycle
rule expires shards on the same schedule as artifacts, so drift sees a rolling
window rather than all history. Neither skews a measurement taken over tens of
queries.

**Two signals, because one cannot say why.** Rising refusals mean something is
wrong, not what. The pairing that resolves it comes straight out of the floor
research above: near-miss questions score *identically* to answerable ones and
are never rejected by the floor, while off-topic questions are.

| refusals | rejections by the floor | diagnosis |
|---|---|---|
| up | unchanged | content gap — the corpus lacks the fact, not the topic |
| up | up | traffic has moved off-topic |
| up | chunks/query down | retrieval or its config regressed |
| unchanged | — | corpus or embedding model changed |

The first calls for writing a document and the third for fixing a bug, so
reporting only "drift detected" would leave the useful part unsaid.

**Characterised, not assumed.** `scripts/validate_drift.py` resamples
`evals/signals.json` — the measured retrieval signal and the real `gpt-4o-mini`
refusal outcome for all 128 golden questions — and is free to re-run:

| window of 100 | 5% contamination | 10% | 20% |
|---|---|---|---|
| near-miss detected | 0% | **100%** | **100%** |
| off-topic detected | 0% | **100%** | **100%** |

False positives on clean traffic are 0–0.4% at every window size, and the named
diagnosis is right 100% of the time for near-miss drift and 99% for off-topic.
Detection sharpened when prompt v2 removed the two false refusals that had been
in the baseline: a reference with less noise in it discriminates better, which
is the argument for refreshing the baseline on every eval rather than pinning
one.
Windows under 30 queries are reported as insufficient rather than scored,
because a rate difference over a handful of queries means nothing.

**Observed in production, not only in simulation.** Against the deployed
service: 32 answerable queries produced a window sitting on the baseline (0%
refusal, mean top score 0.546 against 0.536, 2.96 chunks against 2.87), and the
24-query window it could see was correctly reported as too small to score rather
than scored anyway. A second window of 40 queries carrying 40% near-miss
contamination was caught:

```
  refusal rate             20.0%        0.0%
  ungrounded rate           0.0%        0.0%
  mean top score           0.533       0.536

  [ALERT] refusal_rate: 20.0% of queries refused against a 0.0% baseline (z=3.64)

  Likely a content gap: questions still look like corpus topics -- they retrieve
  normally and score normally -- and are being refused anyway.
```

That is the whole chain closing: the floor research predicted near-miss traffic
would score *identically* to answerable traffic, the offline characterisation
said refusal rate would therefore have to carry the detection, and on real
traffic the similarity distribution did not move (0.533 against 0.536) while
refusals did, and the diagnosis named the right cause unprompted.

Two honest caveats. Those false-positive figures are a lower bound: the null
windows are resampled from the same cases the baseline is built from, so they
are more alike than real traffic would be. And **this detects change, not
badness** — traffic legitimately moving to new topics looks exactly like
traffic degrading, because without labels those *are* the same observation. A
finding is a prompt to go and read the questions, not a verdict. Nothing here
can see a fluent, confident, wrong answer: it arrives with the same similarity
scores as a right one.

### Prompt versioning

The prompts were string constants inside `pipeline.py`, which made them
invisible to everything MLflow recorded. Two eval runs three weeks apart could
differ by four points of accuracy with nothing in the tracking data to say the
prompt had been rewritten in between. The prompt is a parameter of this system
in exactly the way `chunk_size` is, and it was the only one not being logged.

The text now lives in `src/app/rag/prompt_templates/*.txt`, and three things
make a change impossible to miss:

- **Content addressing.** Each template hashes to a fingerprint, and the set
  hashes to one combined fingerprint. Unlike a version number a human
  maintains, a hash cannot be forgotten.
- **A lock file.** `prompts.lock.json` pins the declared version to those
  hashes. Editing a template without re-pinning fails `make test`:

  ```
  AssertionError: prompt templates have drifted from the lock. If the
  change is intended, run: make prompts-lock VERSION=<next version>
  ```

  That is the mechanism, not a side effect — it converts a silent prompt
  change into a reviewable one.
- **`-dirty`.** If the files and the lock disagree at runtime, the version
  logged is `v1-dirty`, borrowing the `git describe` convention. A run is
  never attributed to a clean version it did not use.

Every `/query` run records `prompt_version` and `prompt_fingerprint` as
params. Eval runs additionally store the full text under `prompts/*.txt` — the
artifact you open when accuracy moved and you need to read what changed. Query
runs deliberately do not: artifacts are the one thing that writes to GCS on the
request path, and the text is already recoverable from git by its fingerprint.
`/ready` reports the version too, so a deployed revision can be matched to the
eval run that measured it.

Substitution is `string.Template` (`$context`) rather than `str.format`,
because prompts acquire JSON examples over time and a literal `{` in a
`.format` template is a `KeyError` on the request path. Placeholders are
declared in code and checked against the file at import, so renaming
`$question` to `$query` fails in CI rather than on a user's request.

Two honest limits. The lock proves the text changed, not that anyone
re-measured it — only `make eval` can say whether new wording is better. And
this versions the prompt, not the model: the same prompt against a new
`gpt-4o-mini` snapshot is a different system, which is what the `chat_model`
param is for.

### Why the floor stops here

The similarity floor was rejecting far fewer out-of-corpus questions than
intended, and raising it cost accuracy. The obvious reading is that the
threshold needs to be smarter. It does not; it needs to be *smaller in scope*.

**The measurement was flattering.** The rejection rate was computed over eight
out-of-corpus questions, six of them plainly unrelated (weather, recipes, AWS,
Azure, Kubernetes, PostgreSQL). Re-measured against 67 negatives — 34 of them
deliberate near-misses, on topics the corpus covers but whose specific fact it
lacks — the old floor rejected **11.9%**, not the 38% the small sample showed.

**Relative signals do not help.** The hypothesis was that an out-of-corpus
question produces a *flat* similarity distribution — nothing standing out —
so a per-query signal would separate what an absolute cut cannot. Measured on
real `text-embedding-3-small` scores, every distributional signal was worse
than plain top-1:

| signal | AUC |
|---|---|
| top-1 (absolute) | 0.716 |
| top-1 − mean(tail) | 0.692 |
| z-score | 0.668 |
| top-1 / mean(tail) | 0.557 |

**Because the failure is not distributional.** Splitting the negatives by kind
shows where every signal dies (0.500 is chance):

| signal | vs. near-miss | vs. far / adjacent / meta / injection |
|---|---|---|
| top-1 | **0.518** | 0.927 |
| top-1 − mean | 0.480 | 0.916 |
| z-score | 0.499 | 0.848 |
| distinct docs in top-4 | 0.446 | 0.573 |

Near-miss questions are *indistinguishable from answerable ones* — top-1
similarity for answerable questions runs 0.328/0.535/0.723 (min/median/max)
and for near-misses 0.318/0.526/0.723.

The retrievals show why, and it is not a bug. "What is Cloud Run's maximum
request timeout?" retrieves `cloud-run-scaling` at 0.531. "How much does Secret
Manager charge beyond the free allowance?" retrieves `secret-manager` at 0.687.
**The retriever is right every time** — it fetches exactly the document a person
would reach for. That document simply never states the fact.

Embedding similarity measures *topical relevance*. Refusing requires *factual
sufficiency*. They are different properties, and no threshold on the first
recovers the second. So the model doing most of the refusing is not a defect to
engineer away: deciding "this passage is about Cloud Run concurrency but does
not state a maximum" is a reading judgement, and geometry does not make it.

**What the floor was retuned to do** is the job it is good at — cheap rejection
of topically unrelated questions, where it separates at 0.927. Paired against
the old configuration on ten held-out splits:

| | answerable | rejected | chunks/query |
|---|---|---|---|
| `min_similarity=0.20` | 97.7% | 11.2% | 3.97 |
| `0.28` + `min_similarity_ratio=0.60` | 97.7% | **20.6%** | **2.88** |

The answerable-rate difference was exactly 0.000 on all ten splits — strictly
dominant, never worse. The ratio is the more useful half and it is a *cost*
lever, not a refusal one: it drops chunks scoring below 60% of the best hit,
because the same 0.45 chunk is padding beside a 0.80 hit and the best evidence
available beside a 0.50 one. No absolute threshold can treat those differently.

Confirmed end to end on 128 cases against real `gpt-4o-mini`: retrieval hit
rate stayed at **100%**, and cost per case fell **30%** ($0.000100 → $0.000070).

**What the model actually does with the questions the floor cannot catch.**
This is the part the division of labour rests on, so it was measured rather
than assumed: 66 of 67 out-of-corpus questions refused, including 33 of 34
near-misses, and 33 of 33 unrelated ones. The single failure is worth reading
in full, because it is the shape of the whole problem:

> **Q:** What is the highest concurrency value Cloud Run accepts?
> **A:** The highest concurrency value Cloud Run accepts is eighty, as stated
> in passage [1].

The corpus says concurrency *defaults* to eighty and never states a maximum.
The retrieved passage is the correct one; the model read a default as a limit.
No similarity threshold would have prevented this, because the passage it
misread is the passage it should have been given.

### Parameter sweeps

MLflow was recording every call but never comparing two configurations, which
made `chunk_size=800, top_k=4` a guess with a changelog. `make sweep` grids
over chunk size, overlap, retrieval depth and similarity floor, logs one run
per configuration under `stage=sweep`, and ranks them.

It is nearly free because it scores **retrieval only** — no generation. 24
configurations cost **$0.0025**, roughly a third of a single full eval.

```bash
make sweep     # ~$0.002, ranks the grid
make eval      # ~$0.007, confirms a winner end-to-end
```

**The sweep's first version was wrong, in an instructive way.** It ranked on
document-level retrieval hit rate and picked `cs=400 ov=0 k=3 floor=0.4` —
100% hit rate, 47% less context per query. Running the full eval on it gave
**92.6% accuracy against the incumbent's 100%**, five false refusals. Both
configurations scored an identical hit rate of 1.000, so the metric being
optimised could not distinguish them at all.

The cause: hit rate asks whether the right *document* appeared. A tighter floor
filtered out the chunk holding each answer while the document stayed
represented by some other chunk. The metric was blind to the thing that
actually determines whether an answer is possible.

So the sweep now leads on **`answerable_rate`** — did any retrieved chunk
actually contain a fact the answer needs. It is keyword presence, still free,
and unlike hit rate it predicts reality: the failed candidate scores 93.3%
against its measured 92.6% accuracy, while the incumbent scores 98.3% against
100%. `tests/test_sweep.py` pins that ranking order.

The outcome of the sweep was that the existing defaults were already at the
frontier. That is a legitimate result — the value was in learning it rather
than assuming it, and in discovering that the similarity floor rejects only
38% of out-of-corpus questions, so the LLM is doing most of the refusing that
the floor was supposed to make unnecessary.

### What the CI gate can and cannot tell you

The offline gate uses a hashed bag-of-words embedder, which has no semantic
content. On this corpus it scores a deterministic **56.7%** — its ceiling, not
a defect, since it cannot match a paraphrased question to a passage sharing few
literal tokens. The real embedder scores 100% on the identical cases.

So the CI floor is set at 0.45: it catches a retrieval **collapse** (broken
chunker, mis-wired store, inverted comparison) and nothing subtler. Retrieval
*quality* is measured only by `make eval`, gated at 0.85. Treating the CI
number as a quality signal would be reading the fake, not the system.

Every run is logged to MLflow under `stage=evaluation`, with a per-case TSV
report and the list of threshold breaches as artifacts, so quality is tracked
across commits rather than being a number someone once saw in a terminal.

**There is no LLM judge.** A judge would cost money per run, make the CI gate
non-deterministic, and inherit the blind spots of the model family it grades.
Keyword scoring is cheap and repeatable — and it cannot catch a fluent answer
that is subtly wrong, which makes this a regression gate, not a correctness
proof.

The harness takes a pipeline rather than building one, so the same code serves
two callers:

```bash
make test    # free, offline. Runs the golden set through the real retrieval
             # path with a deterministic fake embedder, and fails the build if
             # retrieval regresses. Runs on every push.

make eval    # SPENDS ~$0.002. Runs against real OpenAI to measure generation
             # quality, then exits non-zero if a threshold is breached.
```

Both log to the same MLflow experiment as production traffic, so eval quality
and live quality are directly comparable.

One thing CI cannot check: refusal accuracy. The fake embedder is a hashed
bag-of-words that scores "weather in Mumbai" at 0.57 against unrelated text on
common-word overlap alone, so asserting refusal there would measure the fake
rather than the system. `make eval` measures it where the similarity floor and
the refusal instruction actually apply.

---

## Spend and abuse ceilings

`terraform destroy` removes every GCP resource and stops **none** of the
OpenAI bill. The service is publicly invokable and holds a real API key, so
two ceilings bound it:

- **`DAILY_BUDGET_USD`** (default `0.25`) — checked *before* any embedding or
  completion call, so the ceiling can be overshot by at most one request.
  ~1,200 gpt-4o-mini queries fit inside it. Exhaustion returns **429** with
  `Retry-After`, not 500: the service is healthy, the caller may simply not
  spend more today.
- **`RATE_LIMIT_PER_MINUTE`** (default `30`) — per client, keyed on a hash of
  the API key rather than an IP, because behind Cloud Run every request
  arrives from Google's front end and `X-Forwarded-For` is caller-controlled.

Both hold state **in process**, so with `max-instances=2` the effective limits
are up to 2x the configured values and reset on scale-to-zero. A shared
counter needs Redis or Firestore; neither is free at this budget. An
approximate cap that fails closed beats a perfect one that costs money to run.

`/health` and `/ready` are never throttled — a 429 on `/health` would make
Cloud Run consider the revision unhealthy. `GET /ready` reports remaining
budget:

```json
{"spend": {"spent_usd": 0.0021, "remaining_usd": 0.2479, "calls": 12},
 "persistence": {"enabled": true, "uri": "gs://.../snapshots/chroma.tar.gz"}}
```

---

## Before your credits expire — 10-day checklist

Run this the day before expiry. `scripts/cost_check.sh` automates most of it.

```bash
PROJECT_ID=my-project ./scripts/cost_check.sh
```

- [ ] **Budget alert exists** — `./scripts/set_budget.sh` (do this first, today)
- [ ] **No Compute Engine VMs** — `gcloud compute instances list` returns nothing
- [ ] **No GKE clusters** — `gcloud container clusters list` returns nothing
- [ ] **No Cloud SQL instances** — `gcloud sql instances list` returns nothing
- [ ] **Cloud Run `min-instances = 0`** — the single most expensive misconfiguration
- [ ] **Cloud Run `max-instances` is set** — bounds a traffic spike
- [ ] **Artifact Registry under 0.5 GB** — cleanup policy attached
- [ ] **GCS bucket under 5 GB and in a US region** — lifecycle rule attached
      (now also holds the Chroma snapshot; still a few MB at demo scale)
- [ ] **Check remaining credits** — [console.cloud.google.com/billing](https://console.cloud.google.com/billing) → Credits
- [ ] **Decide: keep or kill.** Everything here is Always Free at demo scale, so
      the service can keep running past expiry. If you would rather be certain:

```bash
PROJECT_ID=my-project ./scripts/teardown.sh     # deletes every billable resource
```

- [ ] **Confirm the in-app spend ceiling is set** — `curl $URL/ready | jq .spend`
      should show `enabled: true`. This is the only bound on the OpenAI bill
      while the service is up; `terraform destroy` does not touch it.
- [ ] **Revoke the OpenAI key** at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
      — billed by OpenAI, unaffected by GCP teardown
- [ ] **Disable billing on the project** for a guaranteed zero bill:
      Console → Billing → *Disable billing*

### What happens when the trial ends

Your project moves to the free tier. Resources **inside** Always Free limits
keep running. Anything beyond them is suspended rather than silently billed —
*unless* you upgraded to a paid account, in which case overages are charged to
your card. If you have not upgraded, the failure mode is downtime, not a bill.

---

## Project layout

```
.
├── src/app/
│   ├── main.py              # FastAPI routes + trace middleware
│   ├── config.py            # env-driven settings, no hardcoded secrets
│   ├── dependencies.py      # DI wiring + optional API-key guard
│   ├── logging_config.py    # Cloud Logging JSON formatter
│   ├── schemas.py           # request/response contracts
│   ├── persistence.py       # Chroma ⇄ GCS snapshots (survives scale-to-zero)
│   ├── rate_limit.py        # per-client token bucket
│   ├── llm/openai_client.py # chat + embedding wrappers (Protocol-based)
│   ├── rag/
│   │   ├── chunking.py      # paragraph-aware splitter
│   │   ├── vectorstore.py   # Chroma, cosine similarity
│   │   ├── prompts.py       # content-addressed prompts + lock file
│   │   ├── prompt_templates/# the prompt text itself, one file each
│   │   └── pipeline.py      # ingest + query orchestration, instrumented
│   ├── monitoring/
│   │   ├── drift.py         # unlabelled quality signals vs. the eval baseline
│   │   └── source.py        # reads recorded /query runs back out of MLflow
│   ├── evaluation/
│   │   ├── dataset.py       # golden-set JSONL loader
│   │   ├── metrics.py       # retrieval/refusal/citation scoring
│   │   └── runner.py        # eval run + MLflow logging + thresholds
│   └── tracking/
│       ├── cost.py          # token → USD/INR, unit-tested
│       ├── spend_guard.py   # daily OpenAI spend ceiling
│       └── mlflow_tracker.py# fail-open MLflow logging
├── evals/                   # golden.jsonl + corpus/, baseline.json, signals.json
├── tests/                   # 185 tests, OpenAI fully mocked
├── terraform/               # AR, GCS, Cloud Run, IAM, WIF, budget
├── scripts/                 # bootstrap, wif, budget, cost_check, teardown, smoke, lock_prompts
├── .github/workflows/       # ci.yml (all branches) + deploy.yml (main)
└── Dockerfile               # multi-stage, non-root, amd64
```

---

## Design notes

**MLflow tracking is fail-open.** Every tracking call is wrapped so that a
locked SQLite file, an unreachable bucket or expired credentials degrade to a
warning — the user still gets their answer. An observability layer that can
take down the service it observes is worse than none. There is a test that
asserts a query succeeds while the tracker raises on every call.

**The test suite never touches the network.** The OpenAI chat and embedding
clients are `Protocol`s with deterministic fakes. The fake embedder is a hashed
bag-of-words rather than random noise, so documents sharing vocabulary really do
score higher — which makes the retrieval assertions meaningful instead of
tautological. `pytest` costs ₹0 and works offline.

**Logs are structured for Cloud Logging.** `severity` drives console levels and
`logging.googleapis.com/trace` groups every line from one request, so you can
click a slow query and see its retrieval and generation timings. Verified: 100%
of the container's stdout is parseable JSON, including uvicorn's own lines.

That last point took a fix worth knowing about. **MLflow calls
`logging.config.dictConfig()`** when it initialises its SQLite store, which
replaces the root handler's formatter and silently reverts the entire process
to plain text. Startup logs still looked correct, so the failure only appeared
once a request had triggered MLflow — at which point Cloud Logging stopped
parsing severity and trace fields altogether. `ensure_structured_logging()`
re-asserts the formatter after any MLflow interaction, and there is a
regression test pinning the behaviour. Two related sources of startup noise are
handled the same way: alembic's ~30 schema-migration lines are suppressed
during backend init, and `GIT_PYTHON_REFRESH=quiet` stops GitPython printing a
13-line warning about the missing `git` binary on every cold start.

**The container runs as non-root** on a slim base, with build tooling confined
to a discarded builder stage. The image is **162 MB compressed** — the figure
that matters, since Artifact Registry bills stored (compressed) layers.

Chroma declares the `kubernetes` client as a hard dependency for server modes
this app never uses (it runs embedded), so the build deletes it: 82 MB of
site-packages for nothing. `onnxruntime` is deliberately *not* pruned despite
also being unused — Chroma constructs its default ONNX embedding function at
class-definition time, so removing it breaks `import chromadb` outright. Since
pruning a declared dependency is exactly the kind of trick that breaks silently
on upgrade, the Dockerfile ends with an import check that **fails the build** if
the pruned image cannot construct the app.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `exec format error` on Cloud Run | arm64 image built on Apple Silicon | `make docker-build` (pins `--platform linux/amd64`) |
| `503` from `/ingest` or `/query` | `OPENAI_API_KEY` missing at startup | Check startup logs; `/health` stays up deliberately so you can see this |
| Query returns "I don't know" | Nothing ingested yet, or the corpus genuinely cannot answer it | Check `/ready` for `collection_size`; refusing an unanswerable question is correct behaviour, see [Why the floor stops here](#why-the-floor-stops-here) |
| Ingested documents disappear | Snapshot restore failed on cold start | See [Durability](#durability-how-state-survives-scale-to-zero); check startup logs for `snapshot restored` |
| `make drift` sees fewer runs than you sent | Runs reach GCS in batches of `MLFLOW_SNAPSHOT_EVERY` (25) | Expected; the newest few are still in the instance's `/tmp` until the next threshold or shutdown |
| WIF auth fails in CI | `attribute_condition` doesn't match the repo | Confirm `github_repo` is exactly `owner/repo` |

---

## Licence

MIT — see [LICENSE](LICENSE).
