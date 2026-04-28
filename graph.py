import re
import json
from typing import TypedDict, List, Literal
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from retriever import retrieve_docs
from hitl import escalate_to_human, log_escalation
from config import LLM_MODEL, GOOGLE_API_KEY, CONFIDENCE_THRESHOLD
from rich.console import Console

console = Console()

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


console = Console()

# ── State Schema ───────────────────────────────────────────────
class GraphState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    context: str
    answer: str
    confidence: float
    route: Literal["answer", "escalate", "unclear"]
    escalation_reason: str
    final_response: str

# ── LLM Setup ─────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.2)

# ── Node 1: Retrieve ──────────────────────────────────────────
def retrieve_node(state: GraphState) -> GraphState:
    console.print("[dim]→ Retrieving relevant documents...[/dim]")
    query = state["query"]
    docs = retrieve_docs(query)
    context = "\n\n".join([d.page_content for d in docs])
    state["retrieved_docs"] = docs
    state["context"] = context
    return state

# ── Node 2: Assess + Route ────────────────────────────────────
import re
import json

def assess_node(state: GraphState) -> GraphState:
    console.print("[dim]→ Assessing query and context...[/dim]")
    query = state["query"]
    context = state["context"]

    if not context.strip():
        state["confidence"] = 0.0
        state["route"] = "escalate"
        state["escalation_reason"] = "No relevant documents found in knowledge base"
        return state

    # Gemma doesn't support system messages — merge everything into one human message
    prompt = f"""You are a query assessor for a customer support system.
Given the user query and context below, respond ONLY with a raw JSON object.
No markdown, no backticks, no explanation — just the JSON.

Required format:
{{"intent": "faq", "confidence": 0.85, "can_answer": true, "escalation_reason": ""}}

Rules:
- intent: one of faq, complaint, technical, billing, out_of_scope
- confidence: float 0.0-1.0 based on how well the context answers the query
- can_answer: true if context has enough info, false otherwise
- escalation_reason: short reason if cannot answer, else empty string

Query: {query}

Context:
{context}

JSON response:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown fences just in case
        raw = re.sub(r"```json|```", "", raw).strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise ValueError(f"No JSON found in: {raw}")

        confidence = float(result.get("confidence", 0.5))
        can_answer = result.get("can_answer", True)
        escalation_reason = result.get("escalation_reason", "")

        console.print(f"[dim]  confidence={confidence:.2f}, can_answer={can_answer}[/dim]")

    except Exception as e:
        console.print(f"[dim]  Assessment error: {e} — defaulting to answer[/dim]")
        confidence = 0.8
        can_answer = True
        escalation_reason = ""

    if can_answer and confidence >= CONFIDENCE_THRESHOLD:
        route = "answer"
        escalation_reason = ""
    else:
        route = "escalate"

    state["confidence"] = confidence
    state["route"] = route
    state["escalation_reason"] = escalation_reason
    return state


def answer_node(state: GraphState) -> GraphState:
    console.print("[dim]→ Generating answer from context...[/dim]")
    query = state["query"]
    context = state["context"]

    answer_prompt = f"""You are a helpful customer support assistant. 
A user has asked you a question. Use the context below to answer it.
Give a clear, helpful, human-readable answer in plain English.
Do NOT return JSON. Do NOT return code. Just answer the question naturally.

Context from knowledge base:
{context}

User's question: {query}

Your answer:"""

    response = llm.invoke([HumanMessage(content=answer_prompt)])
    answer = response.content.strip()

    # Safety check — if Gemma still returns JSON, extract and answer directly
    if answer.strip().startswith("{"):
        answer = "Based on the knowledge base: " + context[:500]

    state["answer"] = answer
    state["final_response"] = answer
    return state

# ── Node 3b: Escalate to Human ────────────────────────────────
def escalate_node(state: GraphState) -> GraphState:
    console.print("[dim]→ Escalating to human agent...[/dim]")
    query = state["query"]
    reason = state["escalation_reason"]
    human_response = escalate_to_human(query, reason)
    log_escalation(query, reason, human_response)
    state["final_response"] = f"[Human Agent]: {human_response}"
    return state

# ── Conditional Edge ──────────────────────────────────────────
def routing_decision(state: GraphState) -> str:
    return state["route"]  # "answer" or "escalate"

# ── Build Graph ───────────────────────────────────────────────
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("assess",   assess_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "assess")
    graph.add_conditional_edges(
        "assess",
        routing_decision,
        {"answer": "answer", "escalate": "escalate"}
    )
    graph.add_edge("answer",   END)
    graph.add_edge("escalate", END)

    return graph.compile()

# Singleton
compiled_graph = build_graph()

def run_query(query: str) -> str:
    initial_state: GraphState = {
        "query": query,
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "confidence": 0.0,
        "route": "answer",
        "escalation_reason": "",
        "final_response": ""
    }
    result = compiled_graph.invoke(initial_state)
    return result["final_response"], result["confidence"], result["route"]