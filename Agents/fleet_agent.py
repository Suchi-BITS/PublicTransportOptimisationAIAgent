# agents/fleet_agent.py
# Fleet Status Agent - monitors vehicle positions, delays, and mechanical status

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.sensor_tools import fetch_fleet_status
from config.settings import transit_config
from data.models import AgentState, VehicleStatus


FLEET_AGENT_SYSTEM_PROMPT = """You are the Fleet Status Monitoring Agent for a public transit optimization system.

Your responsibilities:
1. Track real-time positions and on-time performance of all vehicles
2. Identify vehicles that are severely delayed and causing service gaps
3. Detect mechanical failures or out-of-service vehicles
4. Assess driver hours to identify shift-end constraints
5. Identify bunching (vehicles running too close together) and gapping (large headway gaps)
6. Determine available reserve capacity for emergency deployment

Fleet management principles:
- On-time performance target: {on_time_target}%
- Vehicles late > 8 minutes require active headway regulation (holding, expressing)
- Vehicle bunching (< 2 min gap): instruct leading vehicle to hold, allow following to catch up
- Gap in service (> 2x scheduled headway): consider short-turn or reserve deployment
- Driver hours > 7.5 on shift: flag for imminent relief, do not assign additional work
- Mechanical status "degraded": monitor closely, prepare replacement
- Mechanical status "out_of_service": remove from duty, deploy reserve if available

Fleet capacity calculation:
- Total buses: {total_buses} | Bus capacity: {bus_capacity} pax
- Total metro trains: {total_metro_trains} | Metro capacity: {metro_capacity} pax
- Aim to keep 8-10% of fleet in reserve for surge deployment

When analyzing fleet data, provide:
- Network-wide on-time performance percentage
- Routes with the worst schedule adherence
- Vehicles needing immediate operational interventions
- Available reserve vehicles for deployment
- Driver fatigue risk assessment"""


def run_fleet_agent(state: AgentState) -> AgentState:
    """
    Fleet status agent node for LangGraph.
    Monitors all vehicles for delays, mechanical issues, and capacity availability.
    """
    print("\n[FLEET AGENT] Polling vehicle status across all active routes...")

    llm = ChatOpenAI(
        model=transit_config.model_name,
        temperature=transit_config.temperature,
        api_key=transit_config.openai_api_key
    )

    all_routes = transit_config.bus_routes + transit_config.metro_lines
    fleet_raw = fetch_fleet_status.invoke({"route_ids": all_routes})
    state.vehicle_statuses = [VehicleStatus(**v) for v in fleet_raw]

    # Compute fleet statistics
    total_vehicles = len(fleet_raw)
    on_time = [v for v in fleet_raw if v["on_time_status"] == "on_time"]
    late = [v for v in fleet_raw if v["on_time_status"] == "late"]
    very_late = [v for v in fleet_raw if v["on_time_status"] == "very_late"]
    out_of_service = [v for v in fleet_raw if v["mechanical_status"] == "out_of_service"]
    degraded = [v for v in fleet_raw if v["mechanical_status"] == "degraded"]
    driver_fatigue_risk = [v for v in fleet_raw if v["driver_hours_on_shift"] > 7.0]

    otp = round(len(on_time) / max(total_vehicles, 1) * 100, 1)

    # Build per-route delay summary
    route_delays = {}
    for v in fleet_raw:
        route = v["route_id"]
        if route not in route_delays:
            route_delays[route] = {"delays": [], "vehicles": 0}
        route_delays[route]["vehicles"] += 1
        if v["delay_minutes"] > 0:
            route_delays[route]["delays"].append(v["delay_minutes"])

    route_summary = []
    for route, data in route_delays.items():
        avg_delay = sum(data["delays"]) / len(data["delays"]) if data["delays"] else 0
        route_summary.append(
            f"{route}: {data['vehicles']} vehicles | avg_delay={avg_delay:.1f}min | "
            f"late_vehicles={len(data['delays'])}"
        )

    prompt_messages = [
        SystemMessage(content=FLEET_AGENT_SYSTEM_PROMPT.format(
            on_time_target=transit_config.on_time_target_percent,
            total_buses=transit_config.total_buses,
            bus_capacity=transit_config.bus_capacity,
            total_metro_trains=transit_config.total_metro_trains,
            metro_capacity=transit_config.metro_capacity
        )),
        HumanMessage(content=f"""
Analyze current fleet status across the entire transit network:

FLEET PERFORMANCE SUMMARY:
- Total vehicles active: {total_vehicles}
- On-time: {len(on_time)} ({otp}%) | Late: {len(late)} | Very late: {len(very_late)}
- Out of service (mechanical): {len(out_of_service)}
- Degraded (mechanical): {len(degraded)}
- Driver fatigue risk (>7h shift): {len(driver_fatigue_risk)}
- Network on-time performance: {otp}% (target: {transit_config.on_time_target_percent}%)

BY ROUTE:
{chr(10).join(route_summary)}

VERY LATE VEHICLES (>8 min delay):
{json.dumps([{
    'vehicle_id': v['vehicle_id'],
    'route_id': v['route_id'],
    'delay_minutes': v['delay_minutes'],
    'location': v['current_location'],
    'next_stop': v['next_stop']
} for v in very_late], indent=2)}

OUT OF SERVICE / DEGRADED VEHICLES:
{json.dumps([{
    'vehicle_id': v['vehicle_id'],
    'route_id': v['route_id'],
    'mechanical_status': v['mechanical_status'],
    'location': v['current_location']
} for v in out_of_service + degraded], indent=2)}

DEMAND AND TRAFFIC CONTEXT:
{state.demand_analysis[:800] if state.demand_analysis else "Not available"}

Provide assessment covering:
1. Routes with critical on-time performance failures requiring intervention
2. Vehicles needing immediate dispatch instructions (expressing, short-turning, holding)
3. Mechanical failures impacting service and replacement vehicle needs
4. Driver fatigue constraints on available operational options
5. Reserve vehicle availability for deployment
6. Bunching or gapping situations requiring regulation
""")
    ]

    response = llm.invoke(prompt_messages)
    state.fleet_analysis = response.content
    state.current_agent = "schedule_optimizer"

    print(f"[FLEET AGENT] Fleet OTP: {otp}%. "
          f"{len(very_late)} vehicles very late, {len(out_of_service)} out of service.")

    return state
