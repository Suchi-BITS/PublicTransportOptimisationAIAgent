# graph/transit_graph.py
# Reactive Agent LangGraph — single-node loop with conditional self-edge.
#
# Topology:
#   START -> reactive_agent_node -> should_rerun? -> reactive_agent_node (loop)
#                                                  -> END
#
# The reactive agent runs, checks if any high-severity triggers remain
# unresolved, and optionally re-runs once more to refine actions.
# In an MVP this is a single pass. The loop guard prevents infinite cycles.

from typing import TypedDict, Any, Optional, List
from langgraph.graph import StateGraph, END

from agents.reactive_agent import run_reactive_agent

MAX_REACTIVE_PASSES = 2


class TransitState(TypedDict, total=False):
    pass_count:           int
    vehicles:             Any
    incidents:            Any
    demand:               Any
    traffic:              Any
    triggers:             Any
    actions:              Any
    network_status:       Optional[str]
    optimisation_report:  Optional[str]


def reactive_node(state: TransitState) -> TransitState:
    state["pass_count"] = state.get("pass_count", 0) + 1
    return run_reactive_agent(state)


def should_rerun(state: TransitState) -> str:
    # Only rerun if high-severity unresolved triggers remain and under pass limit
    if state.get("pass_count", 0) >= MAX_REACTIVE_PASSES:
        return "end"
    triggers = state.get("triggers", [])
    if any(t["severity"] == "high" for t in triggers):
        return "rerun"
    return "end"


def build_transit_graph():
    g = StateGraph(TransitState)
    g.add_node("reactive_agent", reactive_node)
    g.set_entry_point("reactive_agent")
    g.add_conditional_edges(
        "reactive_agent",
        should_rerun,
        {"rerun": "reactive_agent", "end": END},
    )
    return g.compile()
