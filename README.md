# 🛒 AxiomCart — Multi-Agent Voice-Enabled Shopping Assistant

AxiomCart is a **multi-agent AI system** built with [LangGraph](https://langchain-ai.github.io/langgraph/) that powers an e-commerce customer assistant. It combines product discovery (via RAG semantic search), sales support (order tracking, escalation), and voice I/O — all orchestrated through a graph of specialised agents that can run in parallel, ask follow-up questions mid-conversation, and synthesize multi-source answers into a single coherent reply.

---

## What We're Building

### The Problem

E-commerce support queries are diverse. A customer might ask _"Do you have wireless headphones under ₹5,000?"_ (product search), _"Where's my order ORD102?"_ (support lookup), or even both at once: _"I ordered some headphones last week and they haven't arrived — also, do you have any Sony alternatives?"_. A single monolithic chatbot struggles to handle all of these well.

### The Solution — A Multi-Agent Architecture

AxiomCart splits the work across **four cooperating nodes** inside a LangGraph `StateGraph`:

```
START → orchestrator ─┬─ product_agent ──→ synthesizer → END
                      └─ support_agent ──↗
```

1. **Orchestrator** — An LLM-powered classifier that reads the customer's message and decides which specialist(s) should handle it. It can dispatch to one agent or both in parallel.

2. **Product Agent** — A subgraph with its own model ⇄ tools loop. It has access to a semantic search tool (`search_product_catalog`) backed by a vector store (RAG). It handles product questions, recommendations, and general chitchat.

3. **Support Agent** — Another subgraph with a model ⇄ tools loop. It can look up orders (`get_order_status`) and escalate to human support (`escalate_to_human`). It features **Human-in-the-Loop (HITL)** — if the customer hasn't provided an order ID, the agent pauses the entire graph via `interrupt()`, asks the user, and resumes once they answer.

4. **Synthesizer** — Merges results when multiple agents were invoked. For single-agent queries it passes the response through; for multi-agent queries it calls the LLM to weave both answers into one coherent reply.

### Key Capabilities

| Capability | How It Works |
|---|---|
| **Parallel agent dispatch** | The orchestrator uses LangGraph's `Send()` to fan out to multiple agents simultaneously. |
| **RAG product search** | Product catalog is embedded into a vector store; the product agent performs similarity search to find relevant items. |
| **Order tracking** | Orders are looked up by ID (e.g. `ORD101`) or customer email from the order database. |
| **Conversation memory** | A `MemorySaver` checkpointer persists state across turns, so multi-turn conversations "just work". |
| **Human escalation** | Creates a support ticket with priority levels and (optionally) sends an email notification via Resend. |
| **Voice I/O** | Microphone input + OpenAI TTS output for a fully spoken conversational experience. |
| **Response synthesis** | When both agents contribute, an LLM merges the results into a single, natural reply. |

---

## Project Structure

```
src/
├── main.py        # CLI entry point — text REPL, voice loop, or single query
├── config.py      # Logger, API keys, OpenAI/LangChain client setup
├── state.py       # LangGraph state definitions (AxiomCartState, WorkerInput, etc.)
├── graph.py       # Builds & compiles the StateGraph with MemorySaver checkpointer
├── nodes.py       # All four graph nodes + agent subgraphs (model ⇄ tools loops)
├── tools.py       # Tool implementations: catalog search, order lookup, escalation
├── data.py        # Static product catalog, order DB, support policies
├── rag.py         # Vector store setup for product catalog
└── voice.py       # VoiceRecorder & VoiceSpeaker wrappers
```


## How the Graph Executes — Step by Step

Here's what happens when a user sends _"My order is delayed and I want to see some headphones"_:

```
1. START
   └─▶ orchestrator_node
        • LLM classifies the query → two tasks:
            - support_agent: "check order status for delay"
            - product_agent: "search headphones"
        • Sets requires_synthesis = true
        • Dispatches via Send() to BOTH agents in parallel

2. PARALLEL EXECUTION
   ┌─▶ product_agent
   │    • Injects PRODUCT_PROMPT as system message
   │    • Runs product_subgraph (model ⇄ tools loop):
   │        model → calls search_product_catalog("headphones")
   │        tools → returns Bose QC45, Sony XM5, boAt Airdopes
   │        model → formats a helpful product summary
   │    • Sends result to synthesizer
   │
   └─▶ support_agent
        • Injects SUPPORT_PROMPT as system message
        • Runs support_subgraph (model ⇄ tools loop):
            model → no tool calls (needs order ID) → interrupt()
            ← HITL: user is asked "Could you provide your order ID?"
            → user responds "ORD102"
            model → calls get_order_status("ORD102")
            tools → returns order details (Delayed, bad weather)
            model → formats empathetic support response
        • Sends result to synthesizer

3. SYNTHESIZER
   └─▶ synthesizer_node
        • Receives 2 agent results
        • LLM merges them into one coherent reply:
          "I'm sorry to hear your order ORD102 is delayed due to
           weather — the new ETA is Feb 15. In the meantime, here
           are some great headphone options: ..."

4. END
   └─▶ final_answer returned to the user
```

---

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- **OpenAI API key** with access to `gpt-4o` and `text-embedding-3-small`
- A microphone + speakers for voice mode

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nsr-19/axiomcart-ai-assistant.git
cd axiomcart-ai-assistant
```

### 2. Install dependencies

With `uv` (recommended):

```bash
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

> **Note for pip users:** Since you're not using `uv`, replace `uv run python` with just `python` in all the usage commands below. For example:
>
> ```bash
> # Text mode
> python -m src.main
>
> # Voice mode
> python -m src.main --voice
> ```
>
> It's also recommended to use a virtual environment:
>
> ```bash
> python -m venv .venv
> source .venv/bin/activate   # Linux/macOS
> .venv\Scripts\activate      # Windows
> pip install -r requirements.txt
> ```

**Core dependencies:**

| Package | Purpose |
|---|---|
| `langgraph` | Graph orchestration, checkpointing, HITL interrupts |
| `langchain-core` | Message types, tool decorator |
| `langchain-openai` | ChatOpenAI LLM + OpenAI Embeddings |
| `openai` | Raw client for TTS / Whisper (voice mode) |
| `python-dotenv` | `.env` file loading |
| `pydantic` | Structured output schemas |

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
OPENAI_API_KEY=sk-...your-key-here...
```


## Usage

The system supports three modes of interaction, all launched from `src.main`.

### Text Mode (default) — Interactive REPL

```bash
uv run python -m src.main
```

```
🛒  AxiomCart Assistant  (type 'quit' to exit)

You: Hi! Do you have any wireless headphones?
Assistant: Hello! Let me search our catalog for you...

You: What's the status of order ORD102?
Assistant: Your order ORD102 for the Bose QC45 Headphones is currently
delayed due to bad weather. The new estimated delivery is Feb 15...

You: quit
Goodbye!
```


Runs one query, prints the answer, and exits. Useful for scripting or quick lookups.

### Voice Mode

```bash
uv run python -m src.main --voice
```

Requires a working microphone. The assistant speaks a welcome greeting, listens for your question (5-second recording windows), processes it through the full agent pipeline, and speaks the answer back via OpenAI TTS. Say "goodbye" to end the session.

---

## Conversation Examples

### Product Discovery

```
You: What phones do you have?
Assistant: We have two great options:
  1. iPhone 15 Pro Max (₹1,59,900) — A17 Pro chip, titanium, 48MP camera
  2. Samsung Galaxy S24 Ultra (₹1,34,999) — 200MP camera, Galaxy AI, S Pen
```

### Order Tracking with HITL

```
You: Where's my order?
🔄 Agent asks: Could you please provide your order ID or email?
You: ORD101
Assistant: Your order ORD101 for the Air Jordan 1 Retro High OG is shipped!
  Estimated delivery: Feb 13, 2026.
```

The graph literally **pauses** at the `interrupt()` call inside the support subgraph, prompts you for the missing info, then **resumes** from exactly where it left off.

### Mixed Query (Both Agents + Synthesis)

```
You: My order ORD102 is late. Also, show me alternatives to what I ordered.
Assistant: I'm sorry about the delay on your ORD102 (Bose QC45). It's
held up due to weather — new ETA is Feb 15. In the meantime, here are
some alternatives:
  • Sony WH-1000XM5 (₹12,999) — industry-leading ANC, 30-hour battery
  • boAt Airdopes 141 (₹1,299) — budget-friendly with 42-hour playback
```

Both agents ran in parallel; the synthesizer merged their outputs.

### Escalation to Human

```
You: I want to speak to a real person about order ORD102.
Assistant: I've created an escalation ticket for you:
  Ticket: ESC-48271
  Priority: HIGH
  A human agent will contact you within 4 hours.
```

---

## Architecture Deep Dive

### Agent Subgraph Pattern

Each specialist agent is not a single function call — it's a **compiled subgraph** with its own internal loop:

```
START → model ─┬─ tools → model  (loop back if more tool calls)
               └─ END             (no tool calls → done)
```

This means the LLM can chain multiple tool calls in sequence (e.g., search for a product, then search again with refined terms) without any custom loop code. LangGraph manages the cycle natively.

### State Management

The `AxiomCartState` TypedDict flows through the entire graph. Key fields:

| Field | Type | Purpose |
|---|---|---|
| `messages` | `list[AnyMessage]` | Full conversation history (additive reducer) |
| `user_query` | `str` | The current user message |
| `tasks` | `list[AgentTask]` | Routing decisions from the orchestrator |
| `requires_synthesis` | `bool` | Whether the synthesizer should merge results |
| `agent_results` | `list[dict]` | Collected outputs from agents (custom reducer) |
| `final_answer` | `str` | The response returned to the user |

The `agent_results_reducer` is notable: sending an empty list resets the field (clears stale results from previous turns), while a non-empty list appends.

### Checkpointing & Memory

The graph is compiled with `MemorySaver`, and every invocation uses a consistent `thread_id`. This means:
- Conversation history persists across turns within a session.
- HITL interrupts can pause and resume across separate `invoke()` calls.
- The orchestrator sees prior context, so follow-up questions work naturally.

### HITL (Human-in-the-Loop) Flow

When the support agent's LLM responds without making any tool calls and no tools have been called yet in the subgraph, it's inferred that the agent is asking the user for missing information. The flow:

1. `support_model()` detects no tool calls + no prior `ToolMessage` in state.
2. Calls `interrupt(response.content)` — this pauses the **entire graph**.
3. The interrupt surfaces in the `__interrupt__` key of the result dict.
4. `AxiomCartAssistant.query()` detects the interrupt, prompts the user (text or voice), and calls `invoke(Command(resume=user_answer))`.
5. The graph resumes inside `support_model()`, which appends the user's answer as a `HumanMessage` and loops back for another LLM call — this time with enough context to call the right tool.

---

## Configuration Reference

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o, embeddings, TTS, and Whisper |

### Models Used

| Model | Usage |
|---|---|
| `gpt-4o` | Orchestrator classification, agent reasoning, response synthesis |
| `text-embedding-3-small` | Product catalog vector embeddings for RAG |
| `whisper-1` | Voice transcription (voice mode only) |
| `tts-1` | Text-to-speech output (voice mode only) |

---



## Troubleshooting

| Issue | Fix |
|---|---|
| `OPENAI_API_KEY is missing` | Copy `.env.example` to `.env` and add your key |
| `ModuleNotFoundError: src.rag` | Create `src/rag.py` — see the Installation section above |
| `ModuleNotFoundError: src.voice` | Create `src/voice.py` or run without `--voice` |
| HITL not triggering | Make sure `MemorySaver` is active (it's the default in `graph.py`) |
| Stale agent results across turns | The `agent_results_reducer` resets on empty list — this is handled automatically by the orchestrator |
| Voice mode: no audio | Check microphone permissions and that `openai` package is installed |

---
