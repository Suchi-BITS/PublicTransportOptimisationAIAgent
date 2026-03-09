# agents/supervisor_agent.py
# Supervisor Agent - orchestrates the agent graph and produces the final optimization report

import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from config.settings import transit_config
from data.models import AgentState, TransitOptimizationSchedule


SUPERVISOR_SYSTEM_PROMPT = """You are the Transit Operations Supervisor AI for {network_name}.

You receive synthesized outputs from all specialized monitoring and planning agents and 
produce a comprehensive operations report for transit management staff.

Your output should enable the on-duty supervisor to:
1. Understand the current state of the network at a glance
2. Know exactly what actions have been taken automatically
3. Identify situations requiring human judgment or escalation
4. Plan the next operational cycle

Report structure:
- Network status (single status word with brief justification)
- Executive summary (3-5 sentences for rapid consumption)
- Critical actions taken and their expected impact
- Situations requiring human supervisor decision
- Performance projections for the planning horizon
- Next monitoring cycle priorities

Network status levels:
- CRITICAL: Service severely disrupted, major passenger impact, escalation required
- DISRUPTED: Significant incidents or demand surges actively managed
- DEGRADED: Below-normal performance, interventions in progress
- NORMAL: Operating within acceptable parameters
- ENHANCED: Above-normal service deployed for events, performing well

Be direct and operational. This report goes to transit control room staff."""


def run_supervisor_agent(state: AgentState) -> AgentState:
    """
    Supervisor agent - both the entry point (routing) and final synthesis node.
    """
    print("\n[SUPERVISOR] Processing state...")

    # Initial entry: route to first monitoring agent
    if not state.demand_analysis and not state.traffic_analysis:
        print("[SUPERVISOR] Initiating transit optimization cycle...")
        state.current_agent = "demand_agent"
        state.iteration_count += 1
        return state

    # Final synthesis pass after all agents have run
    print("[SUPERVISOR] All agents complete. Synthesizing final optimization report...")

    llm = ChatOpenAI(
        model=transit_config.model_name,
        temperature=transit_config.temperature,
        api_key=transit_config.openai_api_key
    )

    # Compute network KPIs
    total_adjustments = len(state.schedule_adjustments)
    total_alerts = len(state.service_alerts)
    critical_adjustments = [a for a in state.schedule_adjustments if a.priority == "critical"]
    vehicles_added = sum(a.vehicles_added for a in state.schedule_adjustments)
    vehicles_removed = sum(a.vehicles_removed for a in state.schedule_adjustments)

    overcrowded_routes = [
        d for d in state.demand_data
        if d.load_factor > transit_config.overcrowding_threshold
    ]
    severe_traffic = [
        t for t in state.traffic_data
        if t.congestion_level in ["heavy", "gridlock"]
    ]
    active_events = len(state.event_data)

    # Fleet OTP calculation
    total_vehicles = len(state.vehicle_statuses)
    on_time_vehicles = [v for v in state.vehicle_statuses if v.on_time_status == "on_time"]
    otp_percent = round(len(on_time_vehicles) / max(total_vehicles, 1) * 100, 1)

    # Determine network status
    network_status = "normal"
    if len(critical_adjustments) > 2 or any(
        a.severity == "critical" for a in state.service_alerts
    ):
        network_status = "critical"
    elif len(overcrowded_routes) > 2 or len(severe_traffic) > 3:
        network_status = "disrupted"
    elif len(overcrowded_routes) > 0 or otp_percent < transit_config.on_time_target_percent:
        network_status = "degraded"
    elif active_events > 0 and total_adjustments > 3:
        network_status = "enhanced"

    # Build report context
    adjustments_summary = []
    for adj in state.schedule_adjustments:
        adjustments_summary.append(
            f"{adj.route_id} [{adj.route_type}]: {adj.adjustment_type} | "
            f"headway={adj.new_headway_minutes}min | "
            f"+{adj.vehicles_added}/-{adj.vehicles_removed} vehicles | "
            f"priority={adj.priority}"
        )

    alerts_summary = []
    for alert in state.service_alerts:
        alerts_summary.append(f"[{alert.severity.upper()}] {alert.headline}")

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT.format(
            network_name=transit_config.network_name
        )),
        HumanMessage(content=f"""
Produce the final transit operations optimization report:

NETWORK METRICS:
- Total routes monitored: {len(transit_config.bus_routes) + len(transit_config.metro_lines)}
- Fleet on-time performance: {otp_percent}% (target: {transit_config.on_time_target_percent}%)
- Overcrowded routes: {len(overcrowded_routes)}
- Active traffic incidents: {len(severe_traffic)} severe corridors
- Active events: {active_events}
- Schedule adjustments executed: {total_adjustments}
- Vehicles added to service: {vehicles_added}
- Vehicles withdrawn: {vehicles_removed}
- Passenger alerts published: {total_alerts}

SCHEDULE ADJUSTMENTS EXECUTED:
{chr(10).join(adjustments_summary) if adjustments_summary else "None"}

PASSENGER ALERTS PUBLISHED:
{chr(10).join(alerts_summary) if alerts_summary else "None"}

DEMAND ANALYSIS:
{state.demand_analysis or "Not available"}

TRAFFIC ANALYSIS:
{state.traffic_analysis or "Not available"}

EVENT ANALYSIS:
{state.event_analysis or "Not available"}

FLEET STATUS:
{state.fleet_analysis or "Not available"}

Network status determined: {network_status.upper()}
Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Planning horizon: {transit_config.planning_horizon_hours} hours

Produce the operations report covering:
1. Network status and one-line justification
2. Executive summary (what is happening, what was done, expected outcome)
3. Top 3 situations requiring human supervisor attention/decision
4. Projected performance improvement from actions taken
5. Next cycle monitoring priorities
""")
    ]

    response = llm.invoke(messages)

    # Estimate passengers served
    estimated_passengers = sum(
        d.current_passengers for d in state.demand_data
    ) * transit_config.planning_horizon_hours

    # Create final optimization plan
    state.optimization_plan = TransitOptimizationSchedule(
        planning_horizon_hours=transit_config.planning_horizon_hours,
        schedule_adjustments=state.schedule_adjustments,
        fleet_deployments=state.fleet_deployments,
        service_alerts=state.service_alerts,
        network_status=network_status,
        estimated_passengers_served=estimated_passengers,
        summary=response.content,
        kpi_projections={
            "on_time_performance_before": otp_percent,
            "on_time_performance_target": transit_config.on_time_target_percent,
            "overcrowded_routes_before": len(overcrowded_routes),
            "overcrowded_routes_after_projection": max(0, len(overcrowded_routes) - vehicles_added),
            "total_adjustments": total_adjustments,
            "alerts_published": total_alerts,
            "vehicles_redeployed": vehicles_added
        }
    )

    state.current_agent = "complete"
    print(f"[SUPERVISOR] Optimization complete. Network status: {network_status.upper()}")

    return state
