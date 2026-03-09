# agents/demand_agent.py
# Passenger Demand Monitoring Agent - analyzes ridership patterns and demand surges

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.sensor_tools import fetch_passenger_demand, get_historical_ridership
from config.settings import transit_config
from data.models import AgentState, PassengerDemandData


DEMAND_AGENT_SYSTEM_PROMPT = """You are the Passenger Demand Monitoring Agent for a public transit optimization system.

Your responsibilities:
1. Analyze real-time passenger load factors across all bus and metro routes
2. Identify overcrowding (load factor > {overcrowding_threshold}) and underutilization (load factor < {underutil_threshold})
3. Compare current demand against historical baselines to detect anomalies
4. Forecast demand changes based on time-of-day patterns
5. Prioritize routes by intervention urgency

Network coverage:
- Bus routes: {bus_routes}
- Metro lines: {metro_lines}
- Planning horizon: {horizon} hours

Load factor interpretation:
- 0.0 - 0.30: Severely underutilized (service reduction candidate)
- 0.30 - 0.60: Normal, efficient operation
- 0.60 - 0.85: Good utilization, monitor closely
- 0.85 - 1.00: Near capacity, priority for intervention
- 1.00+: Overcrowded, passengers being left behind (critical)

Demand index (current vs historical baseline):
- < 0.7: Significantly below normal (consider service reduction)
- 0.7 - 1.3: Normal range
- 1.3 - 1.8: Elevated demand (increase frequency)
- 1.8+: Major surge (deploy emergency capacity)

Your analysis must:
- Rank all routes from most critical to least critical
- Identify specific stations with excessive wait times
- Flag forecast demand surges that require proactive scheduling
- Distinguish between peak-hour patterns vs genuine anomalies"""


def run_demand_agent(state: AgentState) -> AgentState:
    """
    Demand monitoring agent node for LangGraph.
    Fetches and analyzes passenger demand across all transit routes.
    """
    print("\n[DEMAND AGENT] Fetching passenger demand data across all routes...")

    llm = ChatOpenAI(
        model=transit_config.model_name,
        temperature=transit_config.temperature,
        api_key=transit_config.openai_api_key
    )

    all_routes = transit_config.bus_routes + transit_config.metro_lines

    # Fetch live demand data
    demand_raw = fetch_passenger_demand.invoke({"route_ids": all_routes})
    state.demand_data = [PassengerDemandData(**d) for d in demand_raw]

    # Identify critical routes for historical comparison
    critical_routes = [
        d["route_id"] for d in demand_raw
        if d["load_factor"] > transit_config.overcrowding_threshold
        or d["load_factor"] < transit_config.underutilization_threshold
    ]

    # Build concise demand table for LLM
    demand_table = []
    for d in demand_raw:
        status = "OVERCROWDED" if d["load_factor"] > 1.0 else (
            "NEAR_CAPACITY" if d["load_factor"] > transit_config.overcrowding_threshold else (
                "UNDERUSED" if d["load_factor"] < transit_config.underutilization_threshold else "NORMAL"
            )
        )
        demand_table.append(
            f"{d['route_id']} ({d['route_type']}): "
            f"load={d['load_factor']:.2f} [{status}] | "
            f"pax={d['current_passengers']}/{d['vehicle_capacity']} | "
            f"demand_index={d['demand_index']:.2f} | "
            f"wait={d['peak_station_wait_minutes']}min at {d['peak_station']} | "
            f"1h_forecast={d['demand_forecast_1h']}pax"
        )

    prompt_messages = [
        SystemMessage(content=DEMAND_AGENT_SYSTEM_PROMPT.format(
            overcrowding_threshold=transit_config.overcrowding_threshold,
            underutil_threshold=transit_config.underutilization_threshold,
            bus_routes=", ".join(transit_config.bus_routes),
            metro_lines=", ".join(transit_config.metro_lines),
            horizon=transit_config.planning_horizon_hours
        )),
        HumanMessage(content=f"""
Analyze current passenger demand across the entire transit network:

REAL-TIME DEMAND DATA (all routes):
{chr(10).join(demand_table)}

OVERCROWDING THRESHOLD: {transit_config.overcrowding_threshold:.0%} load factor
UNDERUTILIZATION THRESHOLD: {transit_config.underutilization_threshold:.0%} load factor
SURGE THRESHOLD: {transit_config.surge_multiplier_threshold:.1f}x baseline demand index

Full demand data:
{json.dumps(demand_raw, indent=2, default=str)}

Provide analysis covering:
1. Routes requiring immediate capacity increase (ranked by urgency)
2. Routes where service can be reduced to redeploy vehicles elsewhere
3. Specific stations with unacceptable wait times
4. Demand forecast assessment for next 1-3 hours
5. Any demand anomalies vs historical baselines
6. Overall network load summary (% of routes overcrowded, underused, normal)
""")
    ]

    response = llm.invoke(prompt_messages)
    state.demand_analysis = response.content
    state.current_agent = "traffic_agent"

    overcrowded = [d for d in demand_raw if d["load_factor"] > transit_config.overcrowding_threshold]
    underused = [d for d in demand_raw if d["load_factor"] < transit_config.underutilization_threshold]
    print(f"[DEMAND AGENT] {len(overcrowded)} routes overcrowded, "
          f"{len(underused)} routes underutilized across {len(all_routes)} total routes.")

    return state
