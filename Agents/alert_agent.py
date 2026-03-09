# agents/alert_agent.py
# Passenger Alert Agent - generates and publishes service alerts for passengers

import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from tools.action_tools import issue_service_alert
from config.settings import transit_config
from data.models import AgentState, ServiceAlert


ALERT_AGENT_SYSTEM_PROMPT = """You are the Passenger Communications Agent for a public transit optimization system.

Your responsibilities:
1. Generate clear, accurate service alerts for passengers across all channels
2. Communicate schedule changes, delays, and disruptions proactively
3. Provide alternative route options where service is disrupted
4. Issue event service notifications for extra service deployments
5. Craft messages appropriate for each communication channel

Alert writing principles:
- Headline: Maximum 80 characters, direct and specific ("Route B4: 10-15 min delays downtown")
- Body: Plain language, actionable information, include alternatives
- Avoid jargon: Write for general public, not operations staff
- Severity calibration:
  * critical: Service suspended or major network disruption (many passengers stranded)
  * major: Significant delays or route changes affecting many passengers
  * minor: Delays < 10 min, single route, limited impact
  * info: Service improvements, extra service notifications, events

Channels and appropriate content:
- app: Full detail, can include maps and alternatives
- website: Comprehensive with context
- station_displays: Short, immediate impact only (< 40 chars ideal)
- social_media: Conversational, include hashtags, empathetic tone
- sms: Ultra brief, critical alerts only

Always issue alerts for:
- Any route with delays > 8 minutes (at minimum: minor severity)
- Active traffic incidents affecting service (major severity if > 15 min impact)
- Extra event services being deployed (info severity, positive framing)
- Fleet redeployments that change passenger experience
- Any service suspension or major route change (critical severity)

Do NOT issue alerts for:
- Normal peak hour demand (expected and not newsworthy)
- Minor schedule variations < 3 minutes
- Internal operational adjustments with no passenger impact"""


def run_alert_agent(state: AgentState) -> AgentState:
    """
    Passenger alert agent node for LangGraph.
    Generates and publishes appropriate service alerts based on all interventions.
    """
    print("\n[ALERT AGENT] Generating passenger communications for service changes...")

    llm = ChatOpenAI(
        model=transit_config.model_name,
        temperature=0.2,  # Slightly higher for natural alert language
        api_key=transit_config.openai_api_key
    )

    llm_with_tools = llm.bind_tools([issue_service_alert])

    # Summarize what passengers need to know about
    adjustments_summary = []
    for adj in state.schedule_adjustments:
        adjustments_summary.append({
            "route": adj.route_id,
            "type": adj.adjustment_type,
            "headway": adj.new_headway_minutes,
            "vehicles_added": adj.vehicles_added,
            "priority": adj.priority,
            "communication_required": adj.passenger_communication_required,
            "reason": adj.reason
        })

    traffic_alerts = [
        t for t in state.traffic_data
        if t.delay_minutes > 5 or t.incident_type not in ["none", None]
    ]

    events_needing_comms = [
        e for e in state.event_data
        if e.estimated_transit_demand > 500
    ]

    messages = [
        SystemMessage(content=ALERT_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Generate passenger-facing service alerts for all significant service changes and disruptions.

SCHEDULE ADJUSTMENTS MADE (requiring passenger communication):
{json.dumps([a for a in adjustments_summary if a['communication_required']], indent=2)}

TRAFFIC INCIDENTS AFFECTING SERVICE:
{json.dumps([{
    'corridor': t.corridor,
    'congestion': t.congestion_level,
    'delay_min': t.delay_minutes,
    'incident': t.incident_type,
    'description': t.incident_description,
    'routes': t.route_ids_affected,
    'clearance': t.estimated_clearance_time
} for t in traffic_alerts], indent=2, default=str)}

EVENTS WITH EXTRA SERVICE BEING DEPLOYED:
{json.dumps([{
    'event': e.event_name,
    'venue': e.venue,
    'start': str(e.start_time),
    'end': str(e.end_time),
    'station': e.nearest_station,
    'transit_demand': e.estimated_transit_demand,
    'routes': e.affected_routes,
    'pattern': e.demand_pattern
} for e in events_needing_comms], indent=2, default=str)}

FLEET STATUS CONTEXT:
{state.fleet_analysis[:500] if state.fleet_analysis else "Not available"}

Current time: {datetime.now().strftime('%H:%M')}
Network: {transit_config.network_name}

Call issue_service_alert for each situation passengers need to know about.
Prioritize: (1) disruptions causing delays, (2) route changes, (3) extra event services.
Group related alerts where appropriate. Write clearly for general public.
""")
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "issue_service_alert":
                result = issue_service_alert.invoke(tool_call["args"])
                args = tool_call["args"]

                state.service_alerts.append(ServiceAlert(
                    alert_id=result.get("action_id", f"ALT-{datetime.now().strftime('%H%M%S')}"),
                    severity=args.get("severity", "info"),
                    alert_type=args.get("alert_type", "general"),
                    affected_routes=args.get("affected_routes", []),
                    affected_stations=args.get("affected_stations", []),
                    headline=args.get("headline", ""),
                    body=args.get("body", ""),
                    alternative_routes=args.get("alternative_routes", []),
                    channels=args.get("channels", ["app", "website"])
                ))

                messages.append(ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tool_call["id"]
                ))

    state.current_agent = "supervisor"
    print(f"[ALERT AGENT] Published {len(state.service_alerts)} passenger alerts.")

    return state
