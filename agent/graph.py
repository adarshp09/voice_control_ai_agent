from typing import TypedDict, Any, Optional
from langgraph.graph import StateGraph, END
from agent.nodes import classify_intent, execute_tool, DEFAULT_STATE


class AgentState(TypedDict, total=False):
    audio_path: Optional[str]
    transcription: str
    intent: str
    intent_params: dict
    confirmation_required: bool
    confirmed: Optional[bool]
    tool_output: str
    error: str
    history: list


def route_after_classify(state: AgentState) -> str:
    """
    After intent classification:
    - If file op and confirmation pending → wait (caller handles UI confirm)
    - Otherwise → execute tool directly
    """
    if state.get("confirmation_required") and state.get("confirmed") is None:
        return "await_confirmation"
    return "execute_tool"


def route_after_confirm(state: AgentState) -> str:
    """After user responds to confirmation dialog."""
    if state.get("confirmed") is True:
        return "execute_tool"
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("execute_tool", execute_tool)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "await_confirmation": END,
            "execute_tool": "execute_tool",
        },
    )

    graph.add_edge("execute_tool", END)

    return graph.compile()


compiled_graph = build_graph()


def run_classification(transcription: str, history: list) -> AgentState:
    """Run only the classification step; returns state."""
    initial = {**DEFAULT_STATE, "transcription": transcription, "history": history}
    return compiled_graph.invoke(initial)


def run_execution(state: AgentState) -> AgentState:
    """Execute the tool directly without re-routing through the graph."""
    from agent.nodes import execute_tool
    return execute_tool(dict(state))
