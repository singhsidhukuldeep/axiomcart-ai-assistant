"""
Agent Tools — the concrete actions agents can take.

Product Discovery (1 tool):
  • search_product_catalog  – RAG semantic search

Sales Support (2 tools):
  • get_order_status        – order lookup by ID or email
  • escalate_to_human       – HITL ticket + optional email via Resend
"""

from __future__ import annotations

import random
import time

from langchain_core.tools import tool

#from src.config import RESEND_API_KEY, get_logger
from src.config import get_logger
from src.data import ORDER_DATABASE
from src.rag import product_vectorstore

logger = get_logger("tools")


# ── helpers ──────────────────────────────────────────────────

def normalise_order_id(raw: str) -> str:
    """Accept 'ORD101', 'ORD-101', 'ord-101', or just '101' → 'ORD101'."""
    upper = raw.upper().strip()
    clean = upper.replace("ORD-", "").replace("ORD", "").strip()
    return f"ORD{clean}"


def lookup_order_by_email(email: str) -> dict | None:
    """Find the first order matching a customer email."""
    email_lower = email.lower().strip()
    for oid, order in ORDER_DATABASE.items():
        if order["customer_email"].lower() == email_lower:
            return {"order_id": oid, **order}
    return None


# def send_email(to: str, subject: str, body: str) -> str:
#     """Send an email via Resend if configured, otherwise simulate."""
#     if not RESEND_API_KEY:
#         logger.info("Email simulated (no RESEND_API_KEY): to=%s subj=%r", to, subject)
#         return f"Email simulated to {to} (Resend not configured)"

    # try:
    #     import resend
    #     resend.api_key = RESEND_API_KEY
    #     result = resend.Emails.send({
    #         "from": "onboarding@resend.dev",
    #         "to": [to],
    #         "subject": f"[AxiomCart] {subject}",
    #         "html": f"<div style='font-family:Arial;max-width:600px'>"
    #                 f"<h2 style='color:#2563eb'>AxiomCart Support</h2><hr>"
    #                 f"<p>{body}</p><hr>"
    #                 f"<p style='color:#6b7280;font-size:12px'>AxiomCart Customer Support</p></div>",
    #     })
    #     email_id = result.get("id", "unknown") if isinstance(result, dict) else "sent"
    #     logger.info("Email sent via Resend: to=%s id=%s", to, email_id)
    #     return f"Email sent to {to} (ID: {email_id})"
    # except Exception as exc:
    #     logger.exception("Resend email failed")
    #     return f"Email failed: {exc}"


# ═══════════════════════════════════════════════════════════
#  PRODUCT DISCOVERY TOOL
# ═══════════════════════════════════════════════════════════

@tool
def search_product_catalog(query: str) -> str:
    """Search the AxiomCart product catalog using semantic search (RAG).

    Args:
        query: natural-language search, e.g. "wireless headphones under 5000"
    """
    logger.info("search_product_catalog  query=%r", query)
    try:
        docs = product_vectorstore.similarity_search(query, k=3)
        if not docs:
            return "No products found matching your query."
        results = "Found the following products:\n\n"
        for i, doc in enumerate(docs, 1):
            results += f"Product {i}:\n{doc.page_content}\n\n"
        return results
    except Exception as exc:
        logger.exception("Catalog search failed")
        return f"Error searching catalog: {exc}"


# ═══════════════════════════════════════════════════════════
#  SALES SUPPORT TOOLS
# ═══════════════════════════════════════════════════════════

@tool
def get_order_status(identifier: str) -> str:
    """Look up the current status of a customer order.

    Args:
        identifier: an order ID (e.g. "ORD101") OR a customer email address
    """
    logger.info("get_order_status  identifier=%r", identifier)

    # Try as email first
    if "@" in identifier:
        match = lookup_order_by_email(identifier)
        if match:
            oid = match["order_id"]
            order = {k: v for k, v in match.items() if k != "order_id"}
        else:
            return f"No order found for email: {identifier}"
    else:
        oid = normalise_order_id(identifier)
        order = ORDER_DATABASE.get(oid)
        if not order:
            return f"Order {oid} not found. Please verify the order ID."

    info = (
        f"Order {oid}:\n"
        f"  Customer : {order['customer_name']} ({order['customer_email']})\n"
        f"  Product  : {order['product']}\n"
        f"  Price    : ₹{order['price']:,}\n"
        f"  Status   : {order['status']}\n"
        f"  Ordered  : {order['order_date']}\n"
        f"  ETA      : {order['estimated_delivery']}"
    )
    if order.get("delay_reason"):
        info += f"\n  Delay    : {order['delay_reason']}"
    return info


@tool
def escalate_to_human(order_id: str, issue_summary: str, priority: str = "normal") -> str:
    """Escalate to a human support agent. Customer details are pulled
    from the order database. An email notification is sent if Resend
    is configured.

    Args:
        order_id:      the related order (e.g. "ORD101")
        issue_summary: brief description of the problem
        priority:      low | normal | high | urgent
    """
    order_id = normalise_order_id(order_id)
    logger.info("escalate_to_human  order_id=%s  priority=%s", order_id, priority)
    order = ORDER_DATABASE.get(order_id)

    customer_name  = order["customer_name"]  if order else "Unknown"
    customer_email = order["customer_email"] if order else "Unknown"

    ticket_id = f"ESC-{random.randint(10000, 99999)}"
    ESCALATION_QUEUE.append({
        "ticket_id": ticket_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "customer_name": customer_name,
        "customer_email": customer_email,
        "order_id": order_id,
        "issue_summary": issue_summary,
        "priority": priority,
        "status": "open",
    })

    response_times = {"urgent": "1 hour", "high": "4 hours", "normal": "24 hours", "low": "48 hours"}

    # Optionally send email notification
    # email_status = ""
    # if customer_email != "Unknown":
    #     email_status = _send_email(
    #         to=customer_email,
    #         subject=f"Support Ticket {ticket_id} — {priority.upper()} Priority",
    #         body=(
    #             f"Hi {customer_name},<br><br>"
    #             f"We've created support ticket <b>{ticket_id}</b> for your issue:<br>"
    #             f"<i>{issue_summary}</i><br><br>"
    #             f"A support agent will contact you within {response_times.get(priority, '24 hours')}.<br>"
    #             f"Thank you for your patience."
    #         ),
    #     )

    return (
        f"Escalation ticket created.\n"
        f"  Ticket   : {ticket_id}\n"
        f"  Priority : {priority.upper()}\n"
        f"  Customer : {customer_name} ({customer_email})\n"
        f"  ETA      : within {response_times.get(priority, '24 hours')}\n"
        # f"  Email    : {email_status or 'No email sent (unknown customer)'}\n\n"
        f"A human agent will follow up shortly."
    )
