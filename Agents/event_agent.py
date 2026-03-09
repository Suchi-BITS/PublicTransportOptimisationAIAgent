# agents/event_agent.py
# Event Impact Agent - analyzes scheduled events and their transit demand implications

import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.sensor_tools import fetch_events_calendar
from config.settings import transit_config
from data.models import AgentState, EventData


EVENT_AGENT_SYSTEM_PROMPT = """You are the Event Impact Analysis Agent for a public transit optimization system.

Your responsibilities:
1. Analyze scheduled events and their expected transit demand impact
2. Determine required service augmentation before, during, and after events
3. Identify which routes need reinforcement for each event
4. Assess timing: when does demand surge start, peak, and dissipate
5. Recommend special services (shuttles, express routes) for major events

Event impact assessment framework:

By transit demand volume:
- < 500 transit pax: Minor impact, increase frequency on affected route by 20%
- 500-2000 transit pax: Moderate impact, add extra trips, extend operating hours
- 2000-5000 transit pax: Major impact, deploy express/shuttle services, coordinate with venue
- 5000+ transit pax: Critical impact, activate event special operations plan

By demand pattern:
- surge_before: Heavy demand in 2hr window before event starts
- surge_after: Massive concentrated demand when event ends (most critical)
- both: Split peak with pre-event arrival spread and post-event concentration
- steady: Continuous elevated demand throughout (e.g., markets, conferences)

Post-event surges are the most operationally challenging because:
- All passengers want service simultaneously within 30-60 minutes of event end
- Heavy load factors make boarding slow, causing cascading delays
- Traffic congestion also peaks simultaneously for bus routes

Special service principles:
- Stadium/arena events: Deploy staging areas with dedicated shuttles
- Concerts: Extend last service time, notify passengers of extra services
- Conferences: Emphasize morning peak reinforcement and evening dispersal
- Festivals: Continuous extra frequency throughout, extend daytime hours"""


def run_event_agent(state: AgentState) -> AgentState:
    """
    Event impact agent node for LangGraph.
    Analyzes scheduled events and determines required transit service adjustments.
    """
    print("\n[EVENT AGENT] Fetching events calendar and assessing transit impact...")

    llm = ChatOpenAI(
        model=transit_config.model_name,
        temperature=transit_config.temperature,
        api_key=transit_config.openai_api_key
    )

    events_raw = fetch_events_calendar.invoke({
        "hours_ahead": transit_config.planning_horizon_hours
    })
    state.event_data = [EventData(**e) for e in events_raw]

    if not events_raw:
        print("[EVENT AGENT] No events found in planning horizon.")
        state.event_analysis = "No scheduled events in the planning horizon require special service."
        state.current_agent = "fleet_agent"
        return state

    # Build event impact summary
    event_summary = []
    for e in events_raw:
        time_until_start = ""
        try:
            start = datetime.fromisoformat(e["start_time"])
            diff = (start - datetime.now()).total_seconds() / 3600
            time_until_start = f"starts in {diff:.1f}h" if diff > 0 else "ONGOING"
        except Exception:
            time_until_start = "unknown"

        event_summary.append(
            f"{e['event_name']} at {e['venue']} ({time_until_start}): "
            f"attendance={e['expected_attendance']:,} | "
            f"transit_demand={e['estimated_transit_demand']:,}pax | "
            f"pattern={e['demand_pattern']} | "
            f"station={e['nearest_station']} | "
            f"routes={e['affected_routes']} | "
            f"special_service={e['requires_special_service']}"
        )

    # Categorize events by impact level
    critical_events = [e for e in events_raw if e["estimated_transit_demand"] > 5000]
    major_events = [e for e in events_raw if 2000 <= e["estimated_transit_demand"] <= 5000]
    moderate_events = [e for e in events_raw if 500 <= e["estimated_transit_demand"] < 2000]

    prompt_messages = [
        SystemMessage(content=EVENT_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Analyze scheduled events and determine required transit service adjustments:

EVENTS IN PLANNING HORIZON ({len(events_raw)} events):
{chr(10).join(event_summary)}

IMPACT BREAKDOWN:
- Critical (5000+ transit pax): {len(critical_events)} events
- Major (2000-5000 transit pax): {len(major_events)} events
- Moderate (500-2000 transit pax): {len(moderate_events)} events

NETWORK CONTEXT:
Available fleet: {transit_config.total_buses} buses, {transit_config.total_metro_trains} metro trains
Bus capacity: {transit_config.bus_capacity} pax/vehicle
Metro capacity: {transit_config.metro_capacity} pax/train

DEMAND ANALYSIS CONTEXT:
{state.demand_analysis or "Not available"}

TRAFFIC ANALYSIS CONTEXT:
{state.traffic_analysis or "Not available"}

Full event data:
{json.dumps(events_raw, indent=2, default=str)}

For each event, provide:
1. Impact severity rating and operational priority
2. Required extra vehicle count and type (bus/metro)
3. Specific service adjustments (timing, routing, frequency)
4. Staging and dispatch plan for post-event rush
5. Passenger communication recommendations
6. Coordination requirements (venue staff, traffic management)
7. Pre-event preparation timeline
""")
    ]

    response = llm.invoke(prompt_messages)
    state.event_analysis = response.content
    state.current_agent = "fleet_agent"

    print(f"[EVENT AGENT] Analyzed {len(events_raw)} events: "
          f"{len(critical_events)} critical, {len(major_events)} major, "
          f"{len(moderate_events)} moderate impact.")

    return state
