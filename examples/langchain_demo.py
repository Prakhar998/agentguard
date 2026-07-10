#!/usr/bin/env python3
"""AgentGuard watching a real LangChain agent loop.

The agent is the standard LangChain manual tool-calling loop; the model is
a scripted fake chat model so the demo runs keyless — swap in ChatOpenAI /
ChatAnthropic and nothing else changes, AgentGuard only sees callbacks.

The scripted model falls into a search -> summarize -> search loop.
AgentGuardCallback watches every callback, and with auto_intervene="halt"
raises AgentGuardHalt out of the run when risk crosses the threshold.

    pip install "agentguard[langchain]"
    python examples/langchain_demo.py
"""

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agentguard import Guard
from agentguard.adapters.langchain import AgentGuardCallback, AgentGuardHalt


# -- tools ------------------------------------------------------------------


@tool
def search(query: str) -> str:
    """Search the web."""
    return "10 results: reasoning loops, tool cascades, context poisoning..."


@tool
def summarize(text: str) -> str:
    """Summarize a document."""
    return "Summary: agents fail via loops, cascades and drift."


TOOLS = {"search": search, "summarize": summarize}


def ai_tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


# A scripted model: two productive turns, then it loops forever
# (FakeMessagesListChatModel cycles back to the start of the list —
# conveniently exactly what a looping agent does).
SCRIPT = [
    ai_tool_call("search", {"query": "LLM agent failure modes"}, "c1"),
    ai_tool_call("summarize", {"text": "the search results"}, "c2"),
    ai_tool_call("search", {"query": "LLM agent failure modes"}, "c3"),
]


def main() -> None:
    model = FakeMessagesListChatModel(responses=SCRIPT)
    handler = AgentGuardCallback(
        Guard(predictors=["loop", "tool_cascade", "budget_drift"]),
        run_id="langchain-demo",
        auto_intervene="halt",
    )
    config = {"callbacks": [handler]}

    messages = [HumanMessage("Research LLM agent failure modes.")]
    print("running LangChain agent under AgentGuard...\n")

    try:
        for turn in range(30):  # the agent's own (too-late) iteration cap
            response = model.invoke(messages, config=config)
            messages.append(response)
            if not response.tool_calls:
                print("agent finished normally")
                break
            for call in response.tool_calls:
                result = TOOLS[call["name"]].invoke(call["args"], config=config)
                messages.append(ToolMessage(result, tool_call_id=call["id"]))
                print(
                    f"turn {turn:>2}  {call['name']:<10} risk={handler.risk:.2f}  "
                    + " ".join(f"{k}={v:.2f}" for k, v in handler.subscores.items())
                )
    except AgentGuardHalt as halt:
        print(f"\n⛔ {halt}")
        print(
            f"AgentGuard stopped the run at step "
            f"{len(halt.watcher.history)} — 30-turn cap never reached."
        )


if __name__ == "__main__":
    main()
