#!/usr/bin/env python3
# main.py — Public Transport Optimizer AI Agent v2
# Architecture: Reactive Agent
#
# Usage:
#   python main.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from config.settings import transit_config


def make_initial_state() -> dict:
    return {
        "pass_count": 0, "vehicles": None, "incidents": None,
        "demand": None, "traffic": None, "triggers": None,
        "actions": [], "network_status": None, "optimisation_report": None,
    }


def print_report(state: dict) -> None:
    print("\n" + "=" * 70)
    print("  TRANSIT NETWORK OPTIMISATION REPORT")
    print(f"  {transit_config.network_name} | {transit_config.city}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    status = state.get("network_status", "unknown")
    print(f"\nNETWORK STATUS: {status.upper()}")

    triggers = state.get("triggers") or []
    if triggers:
        print(f"\nTRIGGERS DETECTED ({len(triggers)}):")
        for t in sorted(triggers, key=lambda x: 0 if x["severity"] == "high" else 1):
            print(f"  [{t['severity'].upper():6}] {t['detail']}")

    actions = state.get("actions") or []
    if actions:
        print(f"\nOPTIMISATION ACTIONS ({len(actions)}):")
        for a in actions:
            print(f"\n  Route: {a['route']} | Severity: {a['severity'].upper()}")
            print(f"  Action: {a['recommended_action'].replace('_', ' ').upper()}")
            print(f"  Rationale: {a['rationale'][:200]}")

    vehicles = state.get("vehicles") or []
    if vehicles:
        on_time = sum(1 for v in vehicles
                      if v["delay_minutes"] <= transit_config.delay_threshold_minutes)
        overc   = sum(1 for v in vehicles
                      if v["occupancy_pct"] > transit_config.occupancy_threshold_percent)
        avg_occ = round(sum(v["occupancy_pct"] for v in vehicles) / len(vehicles), 1)
        print(f"\nFLEET SUMMARY:")
        print(f"  {len(vehicles)} vehicles | On-time: {on_time} | "
              f"Overcrowded: {overc} | Avg occupancy: {avg_occ}%")

    print(f"\n{transit_config.disclaimer}")
    print("=" * 70)


def main():
    from agents.base import _demo_mode
    print("=" * 70)
    print("  PUBLIC TRANSPORT OPTIMIZER AI AGENT v2")
    print("  Architecture: Reactive Agent")
    print(f"  Network: {transit_config.network_name}")
    print(f"  Mode: {'DEMO (no API key)' if _demo_mode() else 'LIVE — GPT-4o'}")
    print("=" * 70)

    try:
        from graph.transit_graph import build_transit_graph
        graph  = build_transit_graph()
        result = graph.invoke(make_initial_state())
    except ImportError:
        print("\n[INFO] LangGraph not installed — running agent directly...")
        from agents.reactive_agent import run_reactive_agent
        result = run_reactive_agent(make_initial_state())

    print_report(result)


if __name__ == "__main__":
    main()
