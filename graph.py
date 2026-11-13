"""
agents/graph.py - the LangGraph multi-agent system.

    START -> planner -> (qa | analyst) -> writer -> reviewer -> END      (chart | report | clarify go straight to reviewer)
                                              ^__ one retry: writer fixes unsupported figures __|

  planner   the planner model alone decides the route and companies (no keyword rules).
  qa        ReAct agent: filing questions (search, calculator, fetch).
  analyst   ReAct agent: performance / benchmarking / "why did it change" (metrics engine, drivers, calculator).
  writer    main model (LLM_MODEL) writes the final answer from the evidence the agents gathered (WRITER=off skips).
  chart     specialist: draws charts with the existing chart tool.
  report    specialist: builds the full financial report with the existing report pipeline (unchanged output).
  reviewer  code audit: every figure in the answer must appear in tool results; one repair round, then removal.

Guardrails: tool allow-lists per agent, call limits, result cap, recursion limit, model timeouts/retries,
per-session memory (checkpointer). Failures are caught in agents/__init__.py and shown as a clear message.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Annotated, TypedDict

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

import model
import reviewer

from . import llm as llmmod
from . import tools as T

ROUTES = ("qa", "analyst", "chart", "report", "clarify")
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "10"))
RETRY = RetryPolicy(max_attempts=2)

TEAM_PROMPT = {
    "qa": ("You answer questions about SEC filings (10-K, 10-Q, 8-K) stored in a database. Use search_filing with the "
           "right intent, fetch filings with ensure_filings only if nothing is stored."),
    "analyst": ("You are a financial analyst. For performance or benchmarking call performance_review; to explain why "
                "numbers moved call explain_drivers; for what is new, removed or reworded versus the previous filing call "
                "what_changed; use search_filing for qualitative context."),
}
COMMON = ("\nRules: facts and numbers come ONLY from tool results; never use outside knowledge for figures; never "
          "calculate in your head - use the calculator for every difference, percentage or ratio. Cite the filing (form "
          "and period) for each fact. Be concise. If data is missing, say so; never guess. This is an analysis aid, not "
          "personalised financial advice.")


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    query: str
    company: str
    route: str
    tickers: list
    answer: str
    issues: list
    retries: int


def _text(msg) -> str:
    c = getattr(msg, "content", "") or ""
    if isinstance(c, list):
        c = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return getattr(model, "_clean_text", lambda s: s)(c).strip()


def _hist(session_id: str) -> str:
    try:
        return model._history_text(session_id)
    except Exception:
        return ""


def _rid(config) -> str:
    return config["configurable"]["run_id"]


# ── planner: the planner MODEL alone decides the route (no keyword rules) ────────────────────────────────────────
_PLANNER_PROMPT = (
    "You are the planner of an SEC-filings assistant. Choose ONE specialist for the user's request and list the companies "
    "(ticker symbols) it is about. Reply with JSON only, no other text: "
    '{"route": "qa|analyst|chart|report|clarify", "tickers": ["NVDA"]}.\n'
    "Routes:\n"
    "- qa: questions about what filings say: risks, business, strategy, legal, competition, cybersecurity, a specific fact.\n"
    "- analyst: financial performance, key metrics, benchmarking or comparing companies, why numbers changed, "
    "what changed versus the previous filing.\n"
    "- chart: the user wants a chart, plot or graph.\n"
    "- report: the user wants a full financial report (a written, downloadable report).\n"
    "- clarify: no company can be identified from the request or the chat history.\n"
    "Use the chat history to resolve follow-ups such as 'and their margins?'.\n"
    "History:\n")


def _plan(q: str, company: str, session_id: str) -> tuple[str, list]:
    """Asks the planner model; one re-ask if its reply is not valid JSON with a valid route. Errors propagate."""
    last = ""
    for _ in range(2):
        resp = llmmod.get_chat("planner", max_tokens=300).invoke([
            SystemMessage(content=_PLANNER_PROMPT + (_hist(session_id) or "(none)")),
            HumanMessage(content=q + (f"\n(user picked company: {company})" if company else ""))])
        last = _text(resp)
        m = re.search(r"\{.*\}", last, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except ValueError:
            data = {}
        if data.get("route") in ROUTES:
            tick = [str(t).upper().strip() for t in (data.get("tickers") or []) if str(t).strip()]
            return data["route"], tick
    raise RuntimeError(f"The planner model did not return a valid route (reply: {last[:120]!r})")


def planner(state: State, config) -> dict:
    run = T.get_run(_rid(config))
    q, company = state["query"], state.get("company") or ""
    route, said = _plan(q, company, run.session_id)
    # company names are also looked up by code (name/ticker lookup, not routing); the model's tickers are checked too
    found = list(model.detect_tickers(q, company or None) or [])
    found = [t for t in dict.fromkeys(found + said) if T._covered(t)]
    if route in ("chart", "report", "analyst") and not found:
        route = "clarify"
    run.tickers = found
    run.trace.append(f"planner->{route}({','.join(found)})")
    return {"route": route, "tickers": found, "retries": state.get("retries", 0)}


def _after_planner(state: State) -> str:
    return state["route"]


# ── ReAct specialists (qa, analyst) ────────────────────────────────────────────────────────────────────────────
CONTEXT_CHARS = int(os.getenv("AGENT_CONTEXT_CHARS", "12000"))


def _fit(msgs: list) -> list:
    """Keeps one agent request small: when the conversation is long, older tool results are cut to a short preview."""
    def size(ms):
        return sum(len(str(getattr(m, "content", "") or "")) for m in ms)
    if size(msgs) <= CONTEXT_CHARS:
        return msgs
    out = list(msgs)
    for i, m in enumerate(out[:-2]):                       # the newest messages stay intact
        if m.__class__.__name__ == "ToolMessage" and len(str(m.content)) > 500:
            out[i] = m.model_copy(update={"content": str(m.content)[:500] + " [...older result shortened]"})
        if size(out) <= CONTEXT_CHARS:
            break
    return out


def _react(team: str):
    tool_list = T.tools_for(team)
    tool_node = ToolNode(tool_list, handle_tool_errors=True)

    def agent_node(state: State, config) -> dict:
        chat = llmmod.get_chat("agent").bind_tools(tool_list)
        hist = _hist(T.get_run(_rid(config)).session_id)
        sys_txt = TEAM_PROMPT[team] + COMMON + f"\nCompanies identified: {', '.join(state.get('tickers') or []) or 'none'}"
        if hist:
            sys_txt += f"\nEarlier in this chat:\n{hist}"
        msgs = _fit([SystemMessage(content=sys_txt)] + list(state["messages"]))
        return {"messages": [chat.invoke(msgs)]}

    def route_after_agent(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "finish"

    def finish(state: State, config) -> dict:
        return {"answer": _text(state["messages"][-1])}

    g = StateGraph(State)
    g.add_node("agent", agent_node, retry_policy=RETRY)
    g.add_node("tools", tool_node)
    g.add_node("finish", finish)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "finish": "finish"})
    g.add_edge("tools", "agent")
    g.add_edge("finish", END)
    return g.compile()


def _specialist(team: str):
    sub = _react(team)

    def node(state: State, config) -> dict:
        msgs = [HumanMessage(content=state["query"] + (f"\n(User picked company: {state['company']})" if state.get("company") else ""))]
        if state.get("issues"):
            msgs.append(HumanMessage(content="Your previous answer had figures that do not appear in any tool result:\n- "
                                     + "\n- ".join(state["issues"]) + "\nFix them using tools (calculator for any calculation) or remove them."))
        out = sub.invoke({"messages": msgs, "query": state["query"], "tickers": state.get("tickers", []),
                          "company": state.get("company", "")},
                         {**config, "recursion_limit": MAX_STEPS * 2 + 2})
        return {"answer": out.get("answer", "")}
    return node


# ── direct specialists (chart, report) ─────────────────────────────────────────────────────────────────────────
def chart_node(state: State, config) -> dict:
    run = T.get_run(_rid(config))
    res = T.call_tool(run, "draw_chart", {"tickers": state["tickers"], "request": state["query"]})
    return {"answer": res}


def report_node(state: State, config) -> dict:
    run = T.get_run(_rid(config))
    res = T.call_tool(run, "build_financial_report", {"tickers": state["tickers"]})
    return {"answer": "" if run.report_md else res}


def clarify_node(state: State, config) -> dict:
    return {"answer": "Which company (or companies) do you mean? Please name them, for example NVIDIA and AMD."}


# ── writer: the main model (LLM_MODEL) writes the final answer from the evidence the agents collected ───────────
WRITER_PROMPT = (
    "You are the senior analyst who writes the final answer. Use ONLY the EVIDENCE below (tool results from SEC "
    "filings and the metrics engine). Do not add any number, date or fact that is not in the evidence; do not calculate "
    "in your head - if a figure is not in the evidence, say it is not available. Give clear, insightful, concise prose: "
    "lead with the answer, explain what it means, cite the filing (form and period) for each fact. Not personalised "
    "financial advice.")


def writer_node(state: State, config) -> dict:
    run = T.get_run(_rid(config))
    draft = state.get("answer", "")
    evidence = "\n\n".join(run.evidence)[:14000]
    if (os.getenv("WRITER", "on") or "on").lower() == "off" or not evidence.strip():
        return {"answer": draft}  # nothing gathered (e.g. a clarifying question) or writer switched off
    note = ""
    if state.get("issues"):
        note = ("\nYour previous answer contained figures that are not in the evidence. Remove or correct them:\n- "
                + "\n- ".join(state["issues"]))
    try:
        resp = llmmod.get_chat("writer").invoke([
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(content=f"QUESTION:\n{state['query']}\n\nEVIDENCE:\n{evidence}\n\n"
                                 f"RESEARCH NOTES (may help structure the answer; same rules apply):\n{draft}{note}")])
        out = _text(resp)
        run.trace.append("writer")
        return {"answer": out or draft}
    except Exception as e:
        model.log.warning("writer model failed (%s) - keeping the agent's own answer", e)
        run.trace.append("writer->skipped")
        return {"answer": draft}


# ── reviewer ───────────────────────────────────────────────────────────────────────────────────────────────────
def reviewer_node(state: State, config) -> dict:
    run = T.get_run(_rid(config))
    ans = state.get("answer", "")
    if run.report_md or state["route"] in ("clarify", "chart") or not ans:
        return {"issues": []}
    bad = reviewer.audit(ans, "\n".join(run.evidence))
    if not bad:
        return {"issues": []}
    if state.get("retries", 0) < 1:
        run.trace.append("reviewer->retry")
        return {"issues": bad, "retries": state.get("retries", 0) + 1}
    run.trace.append("reviewer->dropped")
    return {"issues": [], "answer": reviewer._drop(ans, bad) or "I could not support the requested figures from the stored filings."}


def _after_review(state: State) -> str:
    if not state.get("issues"):
        return "done"
    return "writer" if state["route"] in ("qa", "analyst") else state["route"]


# ── assemble ───────────────────────────────────────────────────────────────────────────────────────────────────
_CHECKPOINTER = MemorySaver()


def build_graph(checkpointer=None):
    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("qa", _specialist("qa"))
    g.add_node("analyst", _specialist("analyst"))
    g.add_node("chart", chart_node)
    g.add_node("report", report_node)
    g.add_node("clarify", clarify_node)
    g.add_node("writer", writer_node, retry_policy=RETRY)
    g.add_node("reviewer", reviewer_node)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", _after_planner, {r: r for r in ROUTES})
    for r in ("qa", "analyst"):
        g.add_edge(r, "writer")
    for r in ("chart", "report", "clarify"):
        g.add_edge(r, "reviewer")
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges("reviewer", _after_review, {"writer": "writer", "chart": "chart", "report": "report",
                                                         "clarify": "clarify", "done": END})
    return g.compile(checkpointer=checkpointer or _CHECKPOINTER)


_GRAPH = None


def run_graph(query: str, session_id: str = "default", company: str | None = None) -> dict:
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    rid = uuid.uuid4().hex
    run = T.new_run(rid, query, session_id)
    try:
        out = _GRAPH.invoke({"query": query, "company": company or "", "retries": 0, "issues": [], "messages": []},
                            {"configurable": {"thread_id": f"{session_id}:{rid}", "run_id": rid}, "recursion_limit": 40})
    finally:
        T.drop_run(rid)
    answer = out.get("answer", "")
    if run.report_md:
        answer = (answer + "\n\n---\n\n" if answer else "") + run.report_md
    if not answer:
        answer = "I could not complete that request. Please try rephrasing it, or name the company and the filing."
    mem = model._memory.setdefault(session_id, [])
    mem.append((query, answer[:1500]))
    model._memory[session_id] = mem[-6:]
    return {"answer": answer, "sources": list(dict.fromkeys(run.sources)),
            "decision": {"tickers": run.tickers, "task": "agents", "reason": "graph: " + " > ".join(run.trace), "trace": run.trace},
            "ingest": [], "charts": run.charts, "report_file": run.report_file}