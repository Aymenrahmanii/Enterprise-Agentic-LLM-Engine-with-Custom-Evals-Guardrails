# NexusAgent-Core — System Documentation & Test Guide

> **Enterprise Agentic LLM Engine** featuring vLLM PagedAttention simulation, Redis VSS Semantic Caching, DFA Logit-Masked Guided Decoding, and Async DeepEval Guardrails.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [How Each Subsystem Works](#2-how-each-subsystem-works)
   - [NexusEngine — Guided LLM Decoder](#21-nexusengine--guided-llm-decoder)
   - [GuidedLogitProcessor — DFA Logit Masking](#22-guidedlogitprocessor--dfa-logit-masking)
   - [SemanticCacheManager — Redis VSS](#23-semanticcachemanager--redis-vss)
   - [RouterAgent — Multi-Tool Dispatcher](#24-routeragent--multi-tool-dispatcher)
   - [DatabaseTool — SQL Guardrails](#25-databasetool--sql-guardrails)
   - [AsyncDeepEvalRunner — Evaluation Pipeline](#26-asyncdeepevalrunner--evaluation-pipeline)
3. [Request Lifecycle (End-to-End)](#3-request-lifecycle-end-to-end)
4. [Streamlit Demo — Test Prompts](#4-streamlit-demo--test-prompts)
5. [FastAPI Backend — Test Prompts (cURL / HTTP)](#5-fastapi-backend--test-prompts-curl--http)
6. [Understanding the Metrics](#6-understanding-the-metrics)
7. [Running Locally](#7-running-locally)

---

## 1. Architecture Overview

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Gateway  /api/v1/agent/query    │
│  ┌──────────────────────────────────────────────┐   │
│  │  Stage 1 — Semantic Cache Query (< 8 ms)     │   │
│  │  SemanticCacheManager  →  Redis VSS HNSW      │   │
│  └──────────────┬───────────────────────────────┘   │
│         HIT ◄───┤───► MISS                          │
│                 │                                    │
│  ┌──────────────▼───────────────────────────────┐   │
│  │  Stage 2 — RouterAgent (Intent Classification)│   │
│  │  GuidedLogitProcessor → NexusEngine           │   │
│  │  Guaranteed RouterDecisionSchema JSON output  │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│  ┌──────────────▼───────────────────────────────┐   │
│  │  Stage 3 — Tool Execution                     │   │
│  │  DatabaseTool (SQL + Guardrails)               │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│  ┌──────────────▼───────────────────────────────┐   │
│  │  Stage 4 — Store to Cache + Async DeepEval   │   │
│  │  AsyncDeepEvalRunner (non-blocking)           │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
    │
    ▼
Structured JSON Response + Latency Telemetry
```

---

## 2. How Each Subsystem Works

### 2.1 NexusEngine — Guided LLM Decoder

**File:** [`src/nexus_agent/core/engine.py`](src/nexus_agent/core/engine.py)

The `NexusEngine` is the central generation engine. It runs in two modes:

| Mode | Trigger | Description |
|------|---------|-------------|
| **Mock Mode** (default) | `mock_mode=True` | Simulates vLLM timing (TTFT ~85ms, inter-token ~11.7ms/token). No GPU required. |
| **vLLM Mode** | `mock_mode=False` | Calls `vllm.AsyncLLMEngine.generate()` with custom logit processors on CUDA. |

**What it does:**
1. Receives a `prompt` and a **target Pydantic schema** (e.g., `RouterDecisionSchema`).
2. Instantiates a `GuidedLogitProcessor` seeded with the schema's compiled FSA.
3. Runs generation (mock or real), measuring **TTFT** and **tokens/sec**.
4. Validates the output JSON against the schema — **guaranteed schema-compliant output**.
5. Returns the parsed dict + telemetry metrics.

**Key guarantee:** The output is _always_ a valid JSON object conforming to the requested Pydantic model. If the model would produce invalid JSON, logit masking prevents it at the token level.

---

### 2.2 GuidedLogitProcessor — DFA Logit Masking

**File:** [`src/nexus_agent/core/guided_decoder.py`](src/nexus_agent/core/guided_decoder.py)

This is the core research component. It implements **token-level constrained decoding** using a Deterministic Finite Automaton (DFA).

**The math:**

At each generation step `t`, the raw logit vector `z_t` over the full vocabulary is intercepted and masked:

```
z̃_{t,i} = z_{t,i}   if token i ∈ A_t (allowed set)
         = −∞         otherwise

P(w_t = i | w_{<t}) = exp(z̃_{t,i}) / Σ_{j ∈ A_t} exp(z̃_{t,j})
```

Where `A_t` = the set of token IDs whose string representation keeps the generated text in a **valid partial JSON state** according to the schema's compiled FSA.

**Pipeline:**
1. `SchemaCompiler.get_compiled_fsa(schema)` → compiles the Pydantic schema into a `CompiledSchemaFSA`.
2. At each step, `compute_allowed_tokens(current_text)` checks which vocabulary tokens produce a valid partial JSON continuation.
3. `process_logits()` zeroes out all disallowed tokens (sets them to `-∞`).
4. `apply_softmax_and_sample()` samples from the remaining allowed distribution.

**Result:** The model _cannot_ produce malformed JSON, wrong field names, or wrong types — it's enforced at the probability distribution level, not post-hoc.

---

### 2.3 SemanticCacheManager — Redis VSS

**File:** [`src/nexus_agent/cache/semantic_cache.py`](src/nexus_agent/cache/semantic_cache.py)

A **vector similarity search cache** that avoids calling the LLM for semantically equivalent queries.

**Embedding pipeline:**
- **Primary:** FastEmbed ONNX model `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors).
- **Fallback (no GPU/ONNX):** Deterministic TF-IDF hash projection into 384-dim space.

**How a cache lookup works:**
1. Incoming prompt is embedded → 384-dim float32 vector.
2. Redis VSS (HNSW index) performs approximate nearest-neighbour search.
3. Cosine similarity is computed against stored vectors.
4. If `similarity ≥ τ = 0.92` → **Cache HIT** → return stored response (< 8ms).
5. If `similarity < 0.92` → **Cache MISS** → route to LLM engine.

**Cache store (on miss):**
- After LLM execution, the `(prompt_vector, response_dict)` pair is stored with a TTL of 86400s (24h).
- Subsequent semantically similar queries hit the cache without calling the LLM.

**Business metric:** `get_token_cost_reduction()` returns the cumulative cache hit ratio — the target is **≥ 65% token cost reduction**.

---

### 2.4 RouterAgent — Multi-Tool Dispatcher

**File:** [`src/nexus_agent/agents/router_agent.py`](src/nexus_agent/agents/router_agent.py)

The `RouterAgent` is a multi-tool agent that classifies user intent and dispatches to the appropriate tool using guided decoding.

**Decision schema** (always guaranteed by the guided decoder):
```json
{
  "intent": "database_query",
  "selected_tool": "sql_database_query",
  "tool_args": { "query": "SELECT * FROM users" },
  "explanation": "User is requesting data from the users table."
}
```

**Dispatch logic:**
1. Constructs a prompt with the user query + available tool names.
2. Calls `NexusEngine.generate_guided(prompt, RouterDecisionSchema)` → guaranteed valid JSON.
3. Applies heuristic overrides for common patterns (e.g., queries containing "users", "select", "metric").
4. Looks up the selected tool in `self.tools` and calls `tool.execute(**tool_args)`.
5. Returns an `AgentResponse` with full telemetry.

**Available tools:** `sql_database_query`

---

### 2.5 DatabaseTool — SQL Guardrails

**File:** [`src/nexus_agent/agents/tools/database_tool.py`](src/nexus_agent/agents/tools/database_tool.py)

A read-only SQL execution tool backed by SQLite (`nexus_demo.db`).

**Demo tables pre-seeded on startup:**

`users` table:
| id | name | role | status |
|----|------|------|--------|
| 1 | Alice Vance | AI Architect | active |
| 2 | Bob Smith | LLMOps Lead | active |
| 3 | Charlie Brown | Data Engineer | inactive |

`metrics` table:
| id | metric_name | value |
|----|-------------|-------|
| 1 | ttft_ms | 112.5 |
| 2 | tokens_per_sec | 88.4 |
| 3 | cache_hit_ratio | 0.68 |

**Guardrail enforcement:** Any query containing `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, or `EXEC` is immediately rejected with a guardrail violation error — _before_ reaching the database.

---

### 2.6 AsyncDeepEvalRunner — Evaluation Pipeline

**File:** [`src/nexus_agent/evals/deepeval_runner.py`](src/nexus_agent/evals/deepeval_runner.py)

A **non-blocking** evaluation pipeline that scores every response without adding latency to the user-facing path.

**How it works:**
- Launched via `asyncio.create_task()` — fires and forgets after each request.
- Runs two metrics concurrently:
  1. **SchemaCorrectnessMetric** — validates structural compliance of the JSON output.
  2. **HallucinationMetric** — checks faithfulness of generated content against context.
- Produces an `EvaluationReport` with scores and a pass/fail flag.
- Aggregates running stats via `get_aggregate_stats()` (avg schema score, avg faithfulness, compliance rate).

**Key design:** Zero impact on response latency — evals run after the response is already sent.

---

## 3. Request Lifecycle (End-to-End)

```
POST /api/v1/agent/query  { "query": "show me all users" }
        │
        ├─ 1. Generate trace_id (OpenTelemetry span opened)
        │
        ├─ 2. SemanticCache.query("show me all users")
        │       ├─ Embed query → 384-dim vector
        │       ├─ Redis HNSW nearest-neighbour search
        │       └─ cosine_sim >= 0.92?
        │               ├─ YES → return cached response  (~8ms total)
        │               └─ NO  → continue ↓
        │
        ├─ 3. RouterAgent.process_query("show me all users")
        │       ├─ NexusEngine.generate_guided(prompt, RouterDecisionSchema)
        │       │       ├─ GuidedLogitProcessor masks invalid tokens at each step
        │       │       └─ Returns validated RouterDecisionSchema JSON (~85ms TTFT)
        │       └─ DatabaseTool.execute("SELECT * FROM users")
        │               ├─ Guardrail check (no destructive keywords)
        │               └─ SQLite query → [{"id":1,"name":"Alice Vance",...}, ...]
        │
        ├─ 4. Store result in SemanticCache (TTL=24h)
        │
        ├─ 5. asyncio.create_task(AsyncDeepEvalRunner.evaluate_async(...))  ← non-blocking
        │
        └─ 6. Return AgentQueryResponse with latency breakdown
```

---

## 4. Streamlit Demo — Test Prompts

The Streamlit app at [streamlit_app.py](streamlit_app.py) demonstrates the **cache layer** with two pathways. Copy-paste these into the **"Enter Agent Query Prompt"** field:

### 🟢 Cache HIT Prompts (< 8ms — triggers Redis VSS path)
These contain the keyword `order` which maps to the simulated cache hit branch:

```
How do I query database for missing orders?
```
```
Show me all pending orders from last week
```
```
Get orders that have not been shipped yet
```
```
List all orders with status MISSING
```
```
Find overdue orders in the system
```

**What you'll see:** TTFT = 8ms, `CACHE HIT (Redis VSS HNSW Index, Cosine Sim = 0.94)`, JSON output with `SQL_EXECUTE` action.

---

### 🔴 Cache MISS Prompts (~85ms — triggers LLM engine path)
Any query _without_ "order" routes through the simulated vLLM engine:

```
What is the current system health status?
```
```
Generate a monthly performance analytics report
```
```
Summarize all active users in the platform
```
```
What is the average token throughput this week?
```
```
Explain the cache hit ratio for today
```
```
List all API endpoints and their latency metrics
```

**What you'll see:** TTFT = 85ms, `CACHE MISS (Executed vLLM Engine)`, JSON output with `API_DISPATCH` action.

---

## 5. FastAPI Backend — Test Prompts (cURL / HTTP)

Start the server first:
```bash
uvicorn nexus_agent.main:app --reload --host 0.0.0.0 --port 8000
```

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

---

### Cache MISS → LLM Execution → Tool Dispatch
```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "show me all users in the system", "bypass_cache": false}'
```

**Expected response shape:**
```json
{
  "trace_id": "abc123...",
  "query": "show me all users in the system",
  "action_type": "database_query",
  "tool_name": "sql_database_query",
  "tool_args": {"query": "SELECT * FROM users"},
  "output": {"success": true, "data": [{"id": 1, "name": "Alice Vance", ...}]},
  "cached": false,
  "similarity_score": 0.0,
  "latency_breakdown_ms": {
    "cache_search_ms": 1.2,
    "engine_execution_ms": 95.4,
    "ttft_ms": 85.0,
    "total_end_to_end_ms": 102.3
  }
}
```

---

### Cache HIT (send the same query twice)
Send the query above a second time — the response will now show `"cached": true` and `total_end_to_end_ms < 10`.

---

### Query the metrics table
```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "fetch all performance metrics from the database"}'
```

---

### Force bypass cache (always hits LLM)
```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "show me all users", "bypass_cache": true}'
```

---

### Trigger SQL Guardrail (destructive query — should be rejected)
```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "delete all users from the database"}'
```

**Expected:** `"error": "Guardrail violation: Destructive operation 'DELETE' is strictly disallowed."`

---

### Pass a custom X-Trace-ID header
```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -H "X-Trace-Id: my-custom-trace-001" \
  -d '{"query": "get all active team members"}'
```

---

## 6. Understanding the Metrics

| Metric | What it means | Target |
|--------|---------------|--------|
| **TTFT (ms)** | Time To First Token — latency of the prefill phase (prompt encoding + first token generation). | < 100ms |
| **tokens/sec** | Decode throughput — how many output tokens generated per second. | > 85 tok/s |
| **cache_search_ms** | Time spent on Redis VSS vector similarity search. | < 8ms |
| **total_end_to_end_ms** | Full wall-clock latency from request receipt to response sent. | < 150ms (miss), < 10ms (hit) |
| **cosine_sim** | Vector similarity score between incoming query and best cached entry. | ≥ 0.92 for HIT |
| **schema_compliance_score** | DeepEval score: did the output conform to the RouterDecisionSchema? | 1.0 (100%) |
| **faithfulness_score** | DeepEval score: is the response grounded and faithful (no hallucination)? | ≥ 0.85 |
| **cache_hit_ratio** | `cache_hits / total_queries` — reflects token cost reduction. | ≥ 0.65 (65%) |

---

## 7. Running Locally

### Prerequisites
- Python 3.11+
- Redis (optional — cache falls back to mock without it)

### Install dependencies
```bash
pip install -e ".[dev]"
```

### Run the FastAPI server
```bash
uvicorn src.nexus_agent.main:app --reload --port 8000
```

### Run the Streamlit demo
```bash
streamlit run streamlit_app.py
```

### Run tests
```bash
pytest tests/ -v --cov=src
```

### Run with Docker
```bash
docker-compose up --build
```
The stack includes the FastAPI service + Redis on port 6379.

---

> **Note on Mock Mode:** By default, `NexusEngine` runs in `mock_mode=True`. This simulates vLLM timing (85ms TTFT, 11.7ms/token) on CPU without requiring a GPU or the vLLM library. To connect a real vLLM GPU backend, set `mock_mode=False` in `DependencyContainer` and ensure `vllm` is installed in your CUDA environment.
