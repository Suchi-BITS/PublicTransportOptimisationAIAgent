# agents/reactive_agent.py
# Public Transport Optimizer — Reactive Agent
#
# ══════════════════════════════════════════════════════════════════════════════
# REACTIVE AGENT PATTERN
# ══════════════════════════════════════════════════════════════════════════════
#
# A reactive agent observes the current world state, compares it against
# threshold conditions, and acts only when a threshold is breached.
# It does NOT plan ahead or maintain a goal tree — it reacts to what it sees.
#
# This is appropriate for transit optimisation because:
#   - Most of the time nothing needs changing (system runs normally)
#   - When a threshold is breached (delay, overcrowding, incident), the
#     response must be fast and targeted — not a full replanning exercise
#   - Actions are local: reroute one vehicle, deploy one extra bus, hold one stop
#
# Reactive loop per invocation:
#   1. OBSERVE  — fetch fleet positions, incidents, demand, traffic
#   2. DETECT   — check all vehicles and routes against threshold rules
#   3. REASON   — for each breach, call LLM to choose the best response
#                 RAG provides transit operations best-practice context
#   4. ACT      — log recommended actions with priority and rationale
#   5. REPORT   — produce a network status summary
#
# ══════════════════════════════════════════════════════════════════════════════

import json
from datetime import datetime
from config.settings import transit_config
from data.simulation import (
    get_fleet_positions, get_incidents,
    get_passenger_demand, get_traffic_conditions,
)
from agents.base import call_llm


# ── Threshold detection ───────────────────────────────────────────────────────

def _detect_triggers(vehicles, incidents, demand, traffic) -> list:
    """
    Scan all observations and return a list of triggered conditions.
    Each trigger describes what breached, how badly, and on which route.
    """
    triggers = []

    for v in vehicles:
        if v["delay_minutes"] > transit_config.delay_threshold_minutes:
            triggers.append({
                "type":      "delay",
                "severity":  "high" if v["delay_minutes"] > 15 else "medium",
                "vehicle":   v["vehicle_id"],
                "route":     v["route_id"],
                "stop":      v["current_stop"],
                "value":     v["delay_minutes"],
                "detail":    f"Vehicle {v['vehicle_id']} is {v['delay_minutes']}min late at {v['current_stop']}",
            })
        if v["occupancy_pct"] > transit_config.occupancy_threshold_percent:
            triggers.append({
                "type":      "overcrowding",
                "severity":  "high" if v["occupancy_pct"] > 100 else "medium",
                "vehicle":   v["vehicle_id"],
                "route":     v["route_id"],
                "stop":      v["current_stop"],
                "value":     v["occupancy_pct"],
                "detail":    f"Vehicle {v['vehicle_id']} at {v['occupancy_pct']}% capacity on {v['route_id']}",
            })

    for inc in incidents:
        triggers.append({
            "type":     "incident",
            "severity": inc["severity"],
            "route":    inc["route_affected"],
            "stop":     inc["stop_affected"],
            "value":    inc["estimated_duration_minutes"],
            "detail":   f"{inc['type'].replace('_',' ').title()} on {inc['route_affected']} at {inc['stop_affected']} — est. {inc['estimated_duration_minutes']}min",
        })

    for d in demand:
        if d["avg_wait_minutes"] > 12:
            triggers.append({
                "type":     "high_wait",
                "severity": "medium",
                "route":    d["route_id"],
                "stop":     d["peak_stop"],
                "value":    d["avg_wait_minutes"],
                "detail":   f"Average wait {d['avg_wait_minutes']}min on {d['route_id']} at {d['peak_stop']}",
            })

    if traffic["overall_congestion"] in ("high", "severe"):
        triggers.append({
            "type":     "network_congestion",
            "severity": "high" if traffic["overall_congestion"] == "severe" else "medium",
            "route":    "network-wide",
            "stop":     "multiple",
            "value":    traffic["avg_speed_kmh"],
            "detail":   f"Network congestion {traffic['overall_congestion'].upper()} — avg speed {traffic['avg_speed_kmh']} km/h",
        })

    return triggers


# ── Action catalogue ──────────────────────────────────────────────────────────

RESPONSE_OPTIONS = {
    "delay": [
        "skip_non_essential_stops",
        "hold_connecting_services",
        "deploy_relief_vehicle",
        "update_passenger_information",
        "alert_downstream_stops",
    ],
    "overcrowding": [
        "deploy_extra_vehicle",
        "increase_service_frequency",
        "advise_passengers_next_service",
        "short_turn_vehicle_for_extra_trip",
    ],
    "incident": [
        "reroute_affected_vehicles",
        "activate_diversion_route",
        "deploy_bridge_service",
        "suspend_route_segment",
        "notify_all_affected_passengers",
    ],
    "high_wait": [
        "increase_frequency",
        "deploy_additional_vehicle",
        "split_route_for_higher_frequency",
    ],
    "network_congestion": [
        "activate_bus_priority_signals",
        "advise_alternate_routes_to_passengers",
        "adjust_headways_network_wide",
    ],
}


# ── Reactive agent core ───────────────────────────────────────────────────────

def run_reactive_agent(state: dict) -> dict:
    """
    Single reactive agent invocation.

    Observe -> Detect -> Reason -> Act -> Report
    """
    print("\n[REACTIVE AGENT] Starting transit network observation...")

    # ── 1. OBSERVE ────────────────────────────────────────────────────────────
    vehicles  = get_fleet_positions()
    incidents = get_incidents()
    demand    = get_passenger_demand()
    traffic   = get_traffic_conditions()

    print(f"  Observed: {len(vehicles)} vehicles, {len(incidents)} incident(s), "
          f"congestion={traffic['overall_congestion']}")

    state["vehicles"]  = vehicles
    state["incidents"] = incidents
    state["demand"]    = demand
    state["traffic"]   = traffic

    # ── 2. DETECT ─────────────────────────────────────────────────────────────
    triggers = _detect_triggers(vehicles, incidents, demand, traffic)

    high_triggers   = [t for t in triggers if t["severity"] == "high"]
    medium_triggers = [t for t in triggers if t["severity"] == "medium"]

    print(f"  Detected: {len(triggers)} trigger(s) — "
          f"{len(high_triggers)} high, {len(medium_triggers)} medium")

    state["triggers"] = triggers

    if not triggers:
        print("  No thresholds breached — network operating normally.")
        state["actions"]         = []
        state["network_status"]  = "normal"
        state["optimisation_report"] = _build_normal_report(vehicles, demand, traffic)
        return state

    # ── 3. REASON ─────────────────────────────────────────────────────────────
    # Group triggers by route for efficient LLM reasoning
    by_route: dict = {}
    for t in sorted(triggers, key=lambda x: 0 if x["severity"] == "high" else 1):
        route = t["route"]
        by_route.setdefault(route, []).append(t)

    actions = []
    for route, route_triggers in by_route.items():
        actions.extend(_reason_for_route(route, route_triggers, vehicles, traffic))

    state["actions"] = actions

    # ── 4. NETWORK STATUS ─────────────────────────────────────────────────────
    if any(t["severity"] == "high" for t in triggers):
        status = "disrupted" if len(high_triggers) >= 2 else "degraded"
    else:
        status = "degraded"
    state["network_status"] = status

    # ── 5. REPORT ─────────────────────────────────────────────────────────────
    state["optimisation_report"] = _build_optimisation_report(
        vehicles, triggers, actions, demand, traffic, status
    )

    print(f"  Actions recommended: {len(actions)} | Status: {status.upper()}")
    return state


def _reason_for_route(route: str, triggers: list, vehicles: list, traffic: dict) -> list:
    """
    Call LLM (or demo) to select the best response action for a route's triggers.
    """
    from agents.base import _demo_mode

    route_vehicles = [v for v in vehicles if v["route_id"] == route]
    trigger_text   = "\n".join(f"  - {t['detail']}" for t in triggers)
    options        = []
    for t in triggers:
        options.extend(RESPONSE_OPTIONS.get(t["type"], []))
    options = list(dict.fromkeys(options))[:6]  # deduplicate, cap at 6

    demo = (
        f"Route {route}: {len(triggers)} trigger(s).\n"
        + "\n".join(
            f"  ACTION [{t['severity'].upper()}]: "
            + (RESPONSE_OPTIONS.get(t["type"], ["monitor"])[0])
            + f" — {t['detail']}"
            for t in triggers
        )
    )

    analysis = call_llm(
        system_prompt=(
            f"You are the Reactive Transit Optimisation Agent for {transit_config.network_name}.\n"
            f"Select the single best action for the current trigger on route {route}.\n"
            f"Be decisive and specific. State the action, the vehicles/stops affected, "
            f"and the expected passenger impact."
        ),
        user_prompt=(
            f"TRIGGERS ON {route}:\n{trigger_text}\n\n"
            f"VEHICLES ON ROUTE: {json.dumps(route_vehicles, indent=2)}\n"
            f"TRAFFIC: congestion={traffic['overall_congestion']}, "
            f"avg_speed={traffic['avg_speed_kmh']} km/h\n\n"
            f"AVAILABLE ACTIONS: {', '.join(options)}\n\n"
            f"Select the best action and explain why."
        ),
        demo_response=demo,
    )

    return [{
        "route":       route,
        "triggers":    [t["type"] for t in triggers],
        "severity":    max((t["severity"] for t in triggers),
                          key=lambda s: {"low": 0, "medium": 1, "high": 2}.get(s, 0)),
        "recommended_action": options[0] if options else "monitor",
        "rationale":   analysis,
        "timestamp":   datetime.now().isoformat(timespec="minutes"),
    }]


def _build_optimisation_report(vehicles, triggers, actions, demand, traffic, status) -> str:
    on_time = sum(1 for v in vehicles if v["delay_minutes"] <= transit_config.delay_threshold_minutes)
    overc   = sum(1 for v in vehicles if v["occupancy_pct"] > transit_config.occupancy_threshold_percent)
    avg_occ = round(sum(v["occupancy_pct"] for v in vehicles) / len(vehicles), 1) if vehicles else 0

    return (
        f"[{'DEMO' if not transit_config.openai_api_key.startswith('sk-') else 'LIVE'}]\n\n"
        f"TRANSIT NETWORK OPTIMISATION REPORT\n"
        f"{transit_config.network_name} | {transit_config.city}\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"NETWORK STATUS: {status.upper()}\n\n"
        f"FLEET SUMMARY:\n"
        f"  Total vehicles monitored : {len(vehicles)}\n"
        f"  On time (within {transit_config.delay_threshold_minutes}min) : {on_time}/{len(vehicles)}\n"
        f"  Overcrowded vehicles     : {overc}/{len(vehicles)}\n"
        f"  Average occupancy        : {avg_occ}%\n"
        f"  Network congestion       : {traffic['overall_congestion'].upper()}\n\n"
        f"TRIGGERS DETECTED ({len(triggers)}):\n"
        + "\n".join(f"  [{t['severity'].upper()}] {t['detail']}" for t in triggers)
        + f"\n\nOPTIMISATION ACTIONS ({len(actions)}):\n"
        + "\n".join(
            f"  [{a['severity'].upper()}] {a['route']}: {a['recommended_action'].replace('_',' ').upper()}"
            for a in actions
        )
        + f"\n\n{transit_config.disclaimer}"
    )


def _build_normal_report(vehicles, demand, traffic) -> str:
    avg_occ = round(sum(v["occupancy_pct"] for v in vehicles) / len(vehicles), 1) if vehicles else 0
    return (
        f"TRANSIT NETWORK STATUS: NORMAL\n"
        f"{transit_config.network_name} | {transit_config.city}\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"All routes operating within normal parameters.\n"
        f"Fleet: {len(vehicles)} vehicles | Avg occupancy: {avg_occ}% | "
        f"Congestion: {traffic['overall_congestion']}\n\n"
        f"{transit_config.disclaimer}"
    )
