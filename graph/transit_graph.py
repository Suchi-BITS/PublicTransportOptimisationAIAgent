# graph/transit_graph.py
# LangGraph StateGraph for the Transit Optimization Agent System

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from data.models import AgentState
from agents.demand_agent import run_demand_agent
from agents.traffic_agent import run_traffic_agent
from agents.event_agent import run_event_agent
from agents.fleet_agent import run_fleet_agent
from agents.schedule_optimizer import run_schedule_optimizer
from agents.alert_agent import run_alert_agent
from agents.supervisor_agent import run_supervisor_agent


def build_transit_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph for transit optimization.

    Graph topology:
    
    supervisor (init)
         |
         v
    demand_agent --> traffic_agent --> event_agent --> fleet_agent
                                                           |
                                                           v
                                                   schedule_optimizer
                                                           |
                                                           v
                                                      alert_agent
                                                           |
                                                           v
                                                   supervisor (synthesis)
                                                           |
                                                           v
                                                         END

    The four monitoring agents (demand, traffic, event, fleet) run sequentially,
    each enriching the shared state with their analyses.
    The optimizer then uses all four analyses to make scheduling decisions.
    The alert agent publishes passenger communications.
    The supervisor synthesizes and produces the final report.
    """
    workflow = StateGraph(dict)

    # Wrap Pydantic-based agent functions for LangGraph dict-based state
    def supervisor_node(state: dict) -> dict:
        result = run_supervisor_agent(AgentState(**state))
        return result.model_dump()

    def demand_node(state: dict) -> dict:
        result = run_demand_agent(AgentState(**state))
        return result.model_dump()

    def traffic_node(state: dict) -> dict:
        result = run_traffic_agent(AgentState(**state))
        return result.model_dump()

    def event_node(state: dict) -> dict:
        result = run_event_agent(AgentState(**state))
        return result.model_dump()

    def fleet_node(state: dict) -> dict:
        result = run_fleet_agent(AgentState(**state))
        return result.model_dump()

    def optimizer_node(state: dict) -> dict:
        result = run_schedule_optimizer(AgentState(**state))
        return result.model_dump()

    def alert_node(state: dict) -> dict:
        result = run_alert_agent(AgentState(**state))
        return result.model_dump()

    # Register all nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("demand_agent", demand_node)
    workflow.add_node("traffic_agent", traffic_node)
    workflow.add_node("event_agent", event_node)
    workflow.add_node("fleet_agent", fleet_node)
    workflow.add_node("schedule_optimizer", optimizer_node)
    workflow.add_node("alert_agent", alert_node)

    # Entry point
    workflow.set_entry_point("supervisor")

    # Supervisor routes to first monitoring agent or END
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["current_agent"],
        {
            "demand_agent": "demand_agent",
            "complete": END
        }
    )

    # Sequential monitoring pipeline
    workflow.add_edge("demand_agent", "traffic_agent")
    workflow.add_edge("traffic_agent", "event_agent")
    workflow.add_edge("event_agent", "fleet_agent")

    # Fleet analysis feeds the optimizer
    workflow.add_edge("fleet_agent", "schedule_optimizer")

    # Optimizer feeds alert generation
    workflow.add_edge("schedule_optimizer", "alert_agent")

    # Alerts done, back to supervisor for final synthesis
    workflow.add_edge("alert_agent", "supervisor")

    # Compile with memory checkpointing for persistence
    memory = MemorySaver()
    compiled = workflow.compile(checkpointer=memory)

    return compiled


def get_graph_description() -> str:
    """Return a text description of the graph topology."""
    return """
    TRANSIT OPTIMIZATION AI AGENT GRAPH
    =====================================

    ENTRY
      |
      v
    [SUPERVISOR] -------> [DEMAND AGENT]
       ^                        |
       |                        v
       |                  [TRAFFIC AGENT]
       |                        |
       |                        v
    [ALERT AGENT]         [EVENT AGENT]
       ^                        |
       |                        v
    [SCHEDULE OPTIMIZER] <-- [FLEET AGENT]

    Final: SUPERVISOR (synthesis) --> END

    Agents and their data feeds:
    -------------------------------------------------------
    Supervisor           Orchestrates flow, synthesizes final report
    Demand Agent         AFC systems, APC counters, load factor analysis
    Traffic Agent        Traffic management APIs, incident feeds, AVL data
    Event Agent          Events calendar, venue feeds, demand forecasting
    Fleet Agent          AVL/GPS positions, CAD system, OBD diagnostics
    Schedule Optimizer   LangChain tool-calling -> frequency/fleet adjustments
    Alert Agent          LangChain tool-calling -> multi-channel passenger alerts
    """
