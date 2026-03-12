"""
Graph node functions.

Architecture:

  Main graph:
    START → orchestrator ─┬─ product_agent ──→ synthesizer → END
                          └─ support_agent ──↗

  Each agent is internally a compiled subgraph with a model ⇄ tools loop:
    START → model ─┬─ tools → model (loop back)
                   └─ END    (no tool calls → done)

Key concepts:
  • Tool nodes: tools are separate graph nodes, not manual loops.
    LangGraph controls the model ⇄ tools cycle natively.
  • Parallel dispatch via Send()
  • HITL via conversation persistence — if an agent needs info
    (e.g. order ID), it simply asks. The answer arrives on the
    next turn with full conversation history.
  • Response synthesis for multi-agent queries
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from src.config import get_logger, llm
from src.data import SUPPORT_POLICIES
from src.state import AxiomCartState, ClassificationResult, WorkerInput
from src.tools import (
    escalate_to_human,
    get_order_status,
    search_product_catalog,
)

logger = get_logger("nodes")


# ── Agent Prompts ────────────────────────────────────────────

PRODUCT_PROMPT = """\
You are the Product Discovery Agent for AxiomCart.

ROLE: Help customers find and learn about products. You also handle
general conversation (greetings, thanks, chitchat).

TOOLS:
  search_product_catalog – semantic search over our product database

GUIDELINES:
- For greetings or general chat, respond warmly without calling tools.
- For product questions, always search the catalog first.
- Highlight key features and prices.
- If a product is out of stock, suggest alternatives.
- If the search returns products the customer has already seen or that
  don't match what they asked for (wrong brand, wrong category, etc.),
  be honest and say we don't currently carry what they're looking for.
  Do NOT present irrelevant products as if they match the request.
- Keep responses concise and helpful.
"""

SUPPORT_PROMPT = f"""\
You are the Sales Support Agent for AxiomCart.

ROLE: Handle order enquiries and escalate issues to human agents.

TOOLS:
  get_order_status   – look up an order by order ID or customer email
  escalate_to_human  – create a ticket for human support (sends email notification)

POLICIES:
{SUPPORT_POLICIES}

GUIDELINES:
- If the customer has NOT provided an order ID or email, you MUST ask
  for it before calling any tools. Say something like: "Could you
  please provide your order ID (e.g. ORD101) or registered email
  address so I can look up your order?"
- Be empathetic and professional.
- Only call escalate_to_human when the customer explicitly asks for
  a human agent OR the issue cannot be resolved.
- After retrieving information, respond directly to the customer.
"""


# ── Tool bindings ────────────────────────────────────────────

product_tools = [search_product_catalog]
product_tools_by_name = {t.name: t for t in product_tools}

sales_tools = [get_order_status, escalate_to_human]
sales_tools_by_name = {t.name: t for t in sales_tools}

product_llm = ChatOpenAI(model="gpt-4o", temperature=0.3).bind_tools(product_tools)
sales_llm   = ChatOpenAI(model="gpt-4o", temperature=0.3).bind_tools(sales_tools)


# ═══════════════════════════════════════════════════════════
#  Agent Subgraphs (model ⇄ tools)
# ═══════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """Minimal state for the model ⇄ tools subgraph loop."""
    messages: Annotated[list[AnyMessage], operator.add]


def should_continue(state: AgentState) -> str:
    """Route after model node: tool_calls → tools, otherwise → END."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Product subgraph ─────────────────────────────────────────

def product_model(state: AgentState) -> dict:
    """Call the product LLM (with tools bound)."""
    response = product_llm.invoke(state["messages"])
    logger.info("[product:model] tool_calls=%s", bool(response.tool_calls))
    return {"messages": [response]}


def product_tools(state: AgentState) -> dict:
    """Execute tool calls from the product LLM."""
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        logger.info("[product:tools] %s(%s)", name, args)
        out = product_tools_by_name[name].invoke(args) if name in product_tools_by_name else f"Unknown tool: {name}"
        results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return {"messages": results}


pb = StateGraph(AgentState)
pb.add_node("model", product_model)
pb.add_node("tools", product_tools)
pb.add_edge(START, "model")
pb.add_conditional_edges("model", should_continue)
pb.add_edge("tools", "model")
product_subgraph = pb.compile()


# ── Support subgraph ─────────────────────────────────────────

def support_model(state: AgentState) -> dict:
    """Call the support LLM. If it asks for info without calling tools,
    use interrupt() to pause the graph and collect user input."""
    response = sales_llm.invoke(state["messages"])
    logger.info("[support:model] tool_calls=%s", bool(response.tool_calls))

    # If no tool calls and no tools have been called yet,
    # the agent is asking for missing info — interrupt for HITL
    if not response.tool_calls:
        any_tools_called = any(isinstance(m, ToolMessage) for m in state["messages"])
        if not any_tools_called:
            logger.info("[support:model] HITL: interrupting to collect user info")
            user_reply = interrupt(response.content)
            logger.info("[support:model] HITL: user replied %r", user_reply)
            return {"messages": [response, HumanMessage(content=str(user_reply))]}

    return {"messages": [response]}


def support_tools(state: AgentState) -> dict:
    """Execute tool calls from the support LLM."""
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        logger.info("[support:tools] %s(%s)", name, args)
        out = sales_tools_by_name[name].invoke(args) if name in sales_tools_by_name else f"Unknown tool: {name}"
        results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return {"messages": results}


def support_should_continue(state: AgentState) -> str:
    """Route after support model node. If the last message is a
    HumanMessage (user answered via HITL interrupt), loop back to model."""
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return "model"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


sb = StateGraph(AgentState)
sb.add_node("model", support_model)
sb.add_node("tools", support_tools)
sb.add_edge(START, "model")
sb.add_conditional_edges("model", support_should_continue)
sb.add_edge("tools", "model")
support_subgraph = sb.compile()


# ── Conversation context helper ──────────────────────────────

def build_context(messages: list[AnyMessage]) -> str:
    """Format prior conversation turns as text for agent context."""
    if not messages:
        return ""
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(f"Customer: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
    if not parts:
        return ""
    return "CONVERSATION SO FAR:\n" + "\n".join(parts) + "\n\n"


# ═══════════════════════════════════════════════════════════
#  NODE 1 — Orchestrator
# ═══════════════════════════════════════════════════════════

def orchestrator_node(state: AxiomCartState) -> Command[Literal["product_agent", "support_agent", "synthesizer"]]:
    """Classify the user query and dispatch to the right agent(s)."""
    user_query = state.get("user_query", "")
    if not user_query and state.get("messages"):
        user_query = state["messages"][-1].content

    logger.info("Orchestrator  query=%r", user_query)

    prompt = (
        f'Analyse this customer query and decide which agent(s) should handle it.\n\n'
        f'QUERY: "{user_query}"\n\n'
        'AGENTS:\n'
        '  product_agent – product searches, recommendations, catalog questions,\n'
        '                  AND general conversation (greetings, thanks, chitchat)\n'
        '  support_agent   – order status, complaints, escalation to human support\n\n'
        'RULES:\n'
        '1. Greetings, chitchat, general questions (hi, hello, thanks, how are you)\n'
        '   → product_agent only\n'
        '2. Product-only queries  → product_agent only\n'
        '3. Order/support queries → support_agent only\n'
        '4. Mixed queries         → BOTH agents, requires_synthesis = true\n'
        '\nIMPORTANT: Only route to support_agent when the query clearly involves\n'
        'an order, complaint, or support issue. When in doubt, use product_agent.\n'
    )

    classifier = llm.with_structured_output(ClassificationResult)
    try:
        classification = classifier.invoke(prompt)
    except Exception:
        logger.exception("Classification failed — defaulting to support_agent")
        classification = ClassificationResult(
            tasks=[], requires_synthesis=False,
            reasoning="Fallback: classification error",
        )

    logger.info("  routing=%s  synthesis=%s",
                [t.agent for t in classification.tasks],
                classification.requires_synthesis)

    targets: list[Send] = []
    for task in classification.tasks:
        targets.append(Send(task.agent, {
            "messages": state.get("messages", []),
            "user_query": user_query,
            "task_description": task.task_description,
        }))

    if not targets:
        targets = [Send("synthesizer", {})]

    return Command(
        update={
            "tasks": classification.tasks,
            "requires_synthesis": classification.requires_synthesis,
            "user_query": user_query,
            "agent_results": [],  # reset stale results from prior turns
        },
        goto=targets,
    )


# ═══════════════════════════════════════════════════════════
#  NODE 2 — Product Agent
# ═══════════════════════════════════════════════════════════

def product_agent(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    """Run the product-discovery agent via its model ⇄ tools subgraph."""
    user_query = state.get("user_query", "")
    task_desc  = state.get("task_description", user_query)
    logger.info("Product Agent  task=%r", task_desc)

    context = build_context(state.get("messages", []))

    result = product_subgraph.invoke({"messages": [
        SystemMessage(content=PRODUCT_PROMPT),
        HumanMessage(content=f"{context}Task: {task_desc}\nCustomer query: {user_query}"),
    ]})

    answer = result["messages"][-1].content

    return Command(
        update={"agent_results": [{"source": "product_discovery", "response": answer}]},
        goto="synthesizer",
    )


# ═══════════════════════════════════════════════════════════
#  NODE 3 — Support Agent
# ═══════════════════════════════════════════════════════════

def support_agent(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    """Run the sales-support agent via its model ⇄ tools subgraph.

    HITL is handled through conversation persistence: if the agent
    needs info (e.g. order ID), it responds with a question. The
    user's answer arrives on the next turn via the message history.
    """
    user_query = state.get("user_query", "")
    task_desc  = state.get("task_description", user_query)
    logger.info("Support Agent  task=%r", task_desc)

    context = build_context(state.get("messages", []))

    result = support_subgraph.invoke({"messages": [
        SystemMessage(content=SUPPORT_PROMPT),
        HumanMessage(content=f"{context}Task: {task_desc}\nCustomer query: {user_query}"),
    ]})

    answer = result["messages"][-1].content

    return Command(
        update={"agent_results": [{"source": "sales_support", "response": answer}]},
        goto="synthesizer",
    )


# ═══════════════════════════════════════════════════════════
#  NODE 4 — Synthesizer
# ═══════════════════════════════════════════════════════════

def synthesizer_node(state: AxiomCartState) -> dict:
    """Merge results from one or more agents into a single user-facing reply."""
    results = state.get("agent_results", [])
    user_query = state.get("user_query", "")

    if not results:
        logger.warning("Synthesizer received no agent results")
        return {"final_answer": "Sorry, I couldn't process that request. Please try again."}

    if len(results) == 1:
        logger.info("Synthesizer  single-agent pass-through")
        return {"final_answer": results[0]["response"]}

    logger.info("Synthesizer  merging %d agent responses", len(results))

    parts = "\n\n".join(
        f"[{r['source'].upper()}]:\n{r['response']}" for r in results
    )
    prompt = (
        f"You are combining responses from multiple specialist agents.\n\n"
        f"CUSTOMER QUERY: {user_query}\n\n"
        f"AGENT RESPONSES:\n{parts}\n\n"
        "Write a single, coherent reply that addresses every part of the "
        "customer's query. Be concise. Speak as 'AxiomCart Assistant'."
    )

    merged = llm.invoke(prompt)
    return {"final_answer": merged.content}