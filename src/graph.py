"""
Build and compile the LangGraph StateGraph.

Graph topology:

  START → orchestrator ─┬─ product_agent ─→ synthesizer → END
                        ├─ support_agent ──↗
                        └──────────────────↗ (fallback)

Each agent is internally a subgraph with a model ⇄ tools loop.
The MemorySaver checkpointer persists conversation history across
turns, enabling multi-turn HITL (agent asks a question on one turn,
user answers on the next).
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import get_logger
from src.nodes import orchestrator_node, product_agent, support_agent, synthesizer_node
from src.state import AxiomCartState

logger = get_logger("graph")


def build_graph() -> StateGraph:
    """Create, wire, and compile the AxiomCart multi-agent graph."""

    builder = StateGraph(AxiomCartState)

    # ── Add nodes ────────────────────────────────────────
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("product_agent", product_agent)
    builder.add_node("support_agent", support_agent)
    builder.add_node("synthesizer", synthesizer_node)

    # ── Add edges ────────────────────────────────────────
    builder.add_edge(START, "orchestrator")
    builder.add_edge("synthesizer", END)

    # Dynamic edges defined in src/nodes.py — every node returns Command(goto=...):
    # Send — a target node plus the exact input dict it will receive
    # Command — updates shared state; its goto tells the engine what to run next
    # L266: orchestrator builds one Send per task (product_agent / support_agent)
    # L273: zero tasks — fallback Send straight to synthesizer
    # L282: Command(goto=targets) — the return that actually dispatches the Sends above
    # L307: product_agent returns Command(goto="synthesizer") with its results
    # L337: support_agent returns Command(goto="synthesizer") with its results

    # ── Compile with checkpointer ────────────────────────
    # MemorySaver persists graph state so that interrupt()-based
    # HITL can pause and resume, and conversation history carries
    # forward between queries.
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    logger.info("Graph compiled  (with MemorySaver for conversation persistence)")
    # Print ASCII graph topology in terminal (requires `grandalf` package)
    try:
        logger.info("Graph structure: ")
        print(graph.get_graph().draw_ascii())
    except ImportError as e:
        logger.info("Error while printing the graph: ")
        logger.debug(e)
    return graph


# Module-level singleton
axiomcart_graph = build_graph()
