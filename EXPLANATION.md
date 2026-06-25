# Written Explanation

## Design

The `StateGraph` separates validation, security checks, extraction, read-only
tool calls, conflict checks, and side effects into explicit nodes. Conditional
edges stop the normal quote path whenever the sender is invalid, injection is
detected, extraction fails, or internal tools disagree.

`thread_id` is the LangGraph checkpoint key and also feeds two deterministic
SHA-256 keys: one for the draft and one for the internal note. State flags avoid
repeating completed nodes after a checkpoint resume. `tools.py` adds a durable
SQLite ledger around the non-idempotent write tools. A completed key returns its
stored result; a pending key is never called again automatically.

Exactly-once execution cannot be proven when a remote non-idempotent API creates
an object and the process loses the response. In that ambiguous case the ledger
keeps the key pending and the graph escalates for manual verification instead
of risking a duplicate. Native server-side idempotency would be the production
solution; the same key is also sent in the `Idempotency-Key` HTTP header.

Read-only CRM, ERP, pricing, and shipping calls use bounded exponential backoff
with jitter. Write calls are not blindly retried because doing so is unsafe.
`MemorySaver` satisfies demo checkpointing; production should use a durable
LangGraph checkpointer such as PostgreSQL.

## Security

The email thread is always untrusted. The whole thread is scanned for known
prompt-injection signals, while quote extraction sees only messages whose
domains are allowed by CRM. Email text never selects graph routes, calls tools,
builds URLs, or changes business rules.

OpenRouter is optional and isolated in `extractor.py`. The model receives only
trusted customer text, a system instruction that treats email instructions as
data, and a strict JSON Schema for five nullable fields. Pydantic rejects extra
or invalid fields. The graph, not the model, decides validation, escalation,
pricing, stock, shipping, drafting, and note creation.

Draft text is built from fixed templates and structured tool results. Pricing
formulas are never passed to the model or included in output. Invalid senders,
tool failures, malformed tool responses, unavailable stock, impossible
delivery, late estimated delivery, and prompt injection are escalated.

## Extraction

When both `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` are set, extraction uses
OpenRouter structured outputs. Without them, a deterministic regex fallback
keeps the project runnable offline. The fallback intentionally requires clear
quantity markers to avoid treating a year as the requested quantity.
