# agents/schedule_optimizer.py
# Schedule Optimizer Agent - produces concrete schedule adjustments using LangChain tool calls

import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from tools.action_tools import adjust_route_frequency, deploy_extra_service, reallocate_fleet
from config.settings import transit_config
from data.models import AgentState, ScheduleAdjustment, FleetDeploymentPlan


SCHEDULE_OPTIMIZER_SYSTEM_PROMPT = """You are the Schedule Optimizer Agent for a public transit optimization system.

You receive comprehensive analyses from four monitoring agents and must make concrete,
executable schedule adjustments using the available tools.

Your decision-making framework:

PRIORITY 1 - IMMEDIATE SAFETY AND CRITICAL OVERCROWDING:
- Routes with load factor > 1.0 (standing room exceeded) AND long wait times
- Post-event surges that will overwhelm current service
- Major traffic incidents blocking key corridors

PRIORITY 2 - PROACTIVE DEMAND MANAGEMENT:
- Routes approaching overcrowding (load factor 0.85-1.0) with rising forecast demand
- Events starting within 2 hours requiring augmented service
- Heavy traffic corridors where schedule padding is needed

PRIORITY 3 - EFFICIENCY OPTIMIZATION:
- Reallocate vehicles from underutilized routes (load factor < 0.3) to busy routes
- Regulate headways to prevent bunching and gaps
- Reduce frequency on routes with consistently low ridership

CONSTRAINT RULES:
- Minimum headway: {min_headway} minutes (do not schedule vehicles closer than this)
- Maximum headway: {max_headway} minutes (no gap larger than this for core routes)
- Available buses: {total_buses} (some already deployed, coordinate with fleet analysis)
- Available metro trains: {metro_trains}
- Always call adjust_route_frequency or deploy_extra_service for each concrete action
- Call reallocate_fleet when moving vehicles between routes
- Multiple tool calls are expected and encouraged - be thorough

For each tool call, provide specific, data-driven reasoning that references
the actual load factors, delay times, and event demands from the analyses."""


def run_schedule_optimizer(state: AgentState) -> AgentState:
    """
    Schedule optimizer agent node for LangGraph.
    Uses all monitoring analyses to produce concrete schedule adjustments via tool calls.
    """
    print("\n[SCHEDULE OPTIMIZER] Synthesizing analyses and generating schedule adjustments...")

    llm = ChatOpenAI(
        model=transit_config.model_name,
        temperature=transit_config.temperature,
        api_key=transit_config.openai_api_key
    )

    tools = [adjust_route_frequency, deploy_extra_service, reallocate_fleet]
    llm_with_tools = llm.bind_tools(tools)

    # Build demand snapshot for optimizer
    demand_snapshot = []
    for d in state.demand_data:
        demand_snapshot.append({
            "route": d.route_id,
            "type": d.route_type,
            "load_factor": d.load_factor,
            "demand_index": d.demand_index,
            "wait_min": d.peak_station_wait_minutes,
            "forecast_1h": d.demand_forecast_1h
        })

    # Build traffic impact snapshot
    traffic_snapshot = []
    for t in state.traffic_data:
        if t.delay_minutes > 2 or t.incident_type not in ["none", None]:
            traffic_snapshot.append({
                "corridor": t.corridor,
                "delay_min": t.delay_minutes,
                "congestion": t.congestion_level,
                "routes": t.route_ids_affected,
                "incident": t.incident_type
            })

    # Build events snapshot
    events_snapshot = []
    for e in state.event_data:
        events_snapshot.append({
            "event": e.event_name,
            "station": e.nearest_station,
            "transit_demand": e.estimated_transit_demand,
            "start": e.start_time.isoformat() if hasattr(e.start_time, 'isoformat') else str(e.start_time),
            "end": e.end_time.isoformat() if hasattr(e.end_time, 'isoformat') else str(e.end_time),
            "pattern": e.demand_pattern,
            "routes": e.affected_routes,
            "special_required": e.requires_special_service
        })

    messages = [
        SystemMessage(content=SCHEDULE_OPTIMIZER_SYSTEM_PROMPT.format(
            min_headway=transit_config.min_headway_minutes,
            max_headway=transit_config.max_headway_minutes,
            total_buses=transit_config.total_buses,
            metro_trains=transit_config.total_metro_trains
        )),
        HumanMessage(content=f"""
Generate concrete schedule adjustments based on the following integrated analysis:

DEMAND SNAPSHOT (all routes):
{json.dumps(demand_snapshot, indent=2)}

TRAFFIC IMPACTS (affected corridors):
{json.dumps(traffic_snapshot, indent=2)}

EVENTS REQUIRING SERVICE:
{json.dumps(events_snapshot, indent=2)}

DEMAND ANALYSIS SUMMARY:
{state.demand_analysis or "Not available"}

TRAFFIC ANALYSIS SUMMARY:
{state.traffic_analysis or "Not available"}

EVENT ANALYSIS SUMMARY:
{state.event_analysis or "Not available"}

FLEET STATUS SUMMARY:
{state.fleet_analysis or "Not available"}

Current time: {datetime.now().strftime('%H:%M')}
Planning horizon: {transit_config.planning_horizon_hours} hours

Call adjust_route_frequency, deploy_extra_service, and reallocate_fleet for each
required intervention. Address all routes with load_factor > 0.85 or < 0.25.
Make multiple tool calls to cover all necessary adjustments.
""")
    ]

    # Agentic loop: allow multiple tool call rounds
    max_rounds = 3
    for round_num in range(max_rounds):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not hasattr(response, "tool_calls") or not response.tool_calls:
            break

        # Process all tool calls in this round
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]

            if tool_name == "adjust_route_frequency":
                result = adjust_route_frequency.invoke(args)
                state.schedule_adjustments.append(ScheduleAdjustment(
                    route_id=args["route_id"],
                    route_type=args["route_type"],
                    adjustment_type=args["adjustment_type"],
                    affected_direction=args.get("direction", "both"),
                    new_headway_minutes=args.get("new_headway_minutes"),
                    vehicles_added=args.get("vehicles_added", 0),
                    vehicles_removed=args.get("vehicles_removed", 0),
                    priority=args.get("priority", "normal"),
                    reason=args.get("reason", ""),
                    passenger_communication_required=args.get("priority") in ["critical", "high"]
                ))

            elif tool_name == "deploy_extra_service":
                result = deploy_extra_service.invoke(args)
                state.schedule_adjustments.append(ScheduleAdjustment(
                    route_id=args["route_id"],
                    route_type=args["route_type"],
                    adjustment_type=args["service_type"],
                    vehicles_added=args.get("num_vehicles", 1),
                    priority="high",
                    reason=args.get("reason", ""),
                    passenger_communication_required=True
                ))

            elif tool_name == "reallocate_fleet":
                result = reallocate_fleet.invoke(args)
                state.fleet_deployments.append(FleetDeploymentPlan(
                    route_id=args["to_route_id"],
                    route_type="bus" if args["to_route_id"].startswith("B") else "metro",
                    current_vehicles=0,
                    recommended_vehicles=args.get("num_vehicles", 1),
                    vehicles_to_add=args.get("num_vehicles", 1),
                    source_routes=[args["from_route_id"]],
                    justification=args.get("reason", ""),
                    expected_load_factor_after=0.7
                ))
                result_str = json.dumps(result)

            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            messages.append(ToolMessage(
                content=json.dumps(result) if isinstance(result, dict) else str(result),
                tool_call_id=tool_call["id"]
            ))

    state.current_agent = "alert_agent"
    print(f"[SCHEDULE OPTIMIZER] Generated {len(state.schedule_adjustments)} schedule adjustments, "
          f"{len(state.fleet_deployments)} fleet redeployments.")

    return state
