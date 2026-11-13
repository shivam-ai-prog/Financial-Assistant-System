"""python -m agents.check - run once before turning AGENT_MODE=graph on. Tests the configured models with the real API:
1) the planner model answers JSON, 2) the agent model can CALL A TOOL (not every free model supports tool calling)."""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()
from langchain_core.tools import tool  # noqa: E402

from agents import llm  # noqa: E402


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def main() -> None:
    print("planner model:", llm.model_name("planner"))
    try:
        r = llm.get_chat("planner", 100).invoke('Reply with JSON only: {"route": "qa"}')
        print("  planner OK:", str(r.content)[:80])
    except Exception as e:
        print("  planner FAILED:", e)
    print("agent model:", llm.model_name("agent"))
    try:
        r = llm.get_chat("agent", 200).bind_tools([add]).invoke("Use the add tool to compute 2+3.")
        print("  tool calling OK" if r.tool_calls else "  NO TOOL CALL - choose another AGENT_MODEL (e.g. a paid/other tool-capable model)")
    except Exception as e:
        print("  agent FAILED:", e)


if __name__ == "__main__":
    main()