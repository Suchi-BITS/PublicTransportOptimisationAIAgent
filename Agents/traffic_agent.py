# agents/traffic_agent.py
# Traffic Monitoring Agent - analyzes road conditions and their impact on bus operations

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.sensor_tools import fetch_traffic_conditions
from config.settings import transit_config
from data.models import AgentState, TrafficConditionData


TRAFFIC_AGENT_SYSTEM_PROMPT = """You are the Traffic Monitoring Agent for a public transit optimization system.

Your responsibilities:
1. Monitor real-time traffic conditions on all bus corridors
2. Calculate schedule impact: how many minutes each route is delayed
3. Identify incidents causing disruption and their expected duration
4. Recommend operational responses: detours, skip-stop, express services
5. Flag corridors where bus operations are severely impacted

Traffic impact on transit operations:
- free_flow: No schedule adjustment needed
- light: Minor delays (1-3 min), monitor only
- moderate: Delays 4-8 min, consider short-turning or expressing
- heavy: Delays 8-15 min, active intervention required, passenger communication needed
- gridlock: Delays 15+ min, emergency protocols, consider suspending affected segments

Incident response protocols:
- accident: Reroute affected buses, deploy bypass service if available
- road_works: Adjust schedule to build in extra running time
- event_closure: Pre-planned detours must be activated
- weather: Reduce speeds, increase headways for safety

Metro lines are NOT affected by road traffic but can be affected by:
- Platform overcrowding causing dwell time extensions
- Signal failures and track incidents
- Emergency situations requiring service suspension

Note: Traffic impacts compound with demand surges. Heavy traffic + overcrowding
is the most critical scenario requiring multi-agent coordination."""


def run_traffic_agent(state: AgentState) -> AgentState:
    """
    Traffic monitoring agent node for LangGraph.
    Fetches traffic data and assesses impact on transit schedule adherence.
    """
    print("\n[TRAFFIC AGENT] Fetching traffic conditions across all corridors...")

    llm = ChatOpenAI(
        model=transit_config.model_name,
        temperature=transit_config.temperature,
        api_key=transit_config.openai_api_key
    )

    corridors = [
        "north_south_arterial",
        "east_west_boulevard",
        "airport_express",
        "downtown_loop",
        "university_corridor",
        "stadium_access",
        "tech_district",
        "westside_connector"
    ]

    traffic_raw = fetch_traffic_conditions.invoke({"corridors": corridors})
    state.traffic_data = [TrafficConditionData(**t) for t in traffic_raw]

    # Build corridor summary for LLM
    corridor_summary = []
    for t in traffic_raw:
        corridor_summary.append(
            f"{t['corridor']}: {t['congestion_level'].upper()} | "
            f"speed={t['current_speed_kmh']:.0f}/{t['free_flow_speed_kmh']:.0f} kmh | "
            f"delay=+{t['delay_minutes']:.1f}min | "
            f"incident={t['incident_type']} | "
            f"routes affected={t['route_ids_affected']}"
        )

    # Find heavily impacted routes
    severe_corridors = [
        t for t in traffic_raw
        if t["congestion_level"] in ["heavy", "gridlock"] or t["delay_minutes"] > 8
    ]

    incident_corridors = [
        t for t in traffic_raw
        if t["incident_type"] not in ["none", None]
    ]

    prompt_messages = [
        SystemMessage(content=TRAFFIC_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Analyze current traffic conditions and their impact on bus transit operations:

CORRIDOR TRAFFIC STATUS:
{chr(10).join(corridor_summary)}

SEVERE CONGESTION CORRIDORS ({len(severe_corridors)}):
{json.dumps([{
    'corridor': t['corridor'],
    'congestion': t['congestion_level'],
    'delay_min': t['delay_minutes'],
    'routes': t['route_ids_affected']
} for t in severe_corridors], indent=2)}

ACTIVE INCIDENTS ({len(incident_corridors)}):
{json.dumps([{
    'corridor': t['corridor'],
    'type': t['incident_type'],
    'description': t['incident_description'],
    'clearance': t['estimated_clearance_time'],
    'routes': t['route_ids_affected']
} for t in incident_corridors], indent=2)}

DEMAND CONTEXT:
{state.demand_analysis or "Demand analysis not yet available"}

Full traffic data:
{json.dumps(traffic_raw, indent=2, default=str)}

Provide assessment covering:
1. Routes with schedule-breaking delays requiring immediate response
2. Incidents requiring detour activation or passenger communication
3. Estimated schedule deviation per affected route
4. Recommended operational responses (detours, expressing, schedule padding)
5. Expected time until conditions normalize
6. Compound risk assessment: routes facing both heavy traffic AND high demand
""")
    ]

    response = llm.invoke(prompt_messages)
    state.traffic_analysis = response.content
    state.current_agent = "event_agent"

    print(f"[TRAFFIC AGENT] {len(severe_corridors)} severe corridors, "
          f"{len(incident_corridors)} active incidents detected.")

    return state
