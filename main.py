# main.py
# Entry point for the AI Public Transportation Optimization Agent System

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.transit_graph import build_transit_graph, get_graph_description
from data.models import AgentState
from config.settings import transit_config


def print_header():
    print("=" * 72)
    print("  AI PUBLIC TRANSPORTATION OPTIMIZATION SYSTEM")
    print(f"  Network: {transit_config.network_name}")
    print(f"  City: {transit_config.city}")
    print(f"  Bus routes: {', '.join(transit_config.bus_routes)}")
    print(f"  Metro lines: {', '.join(transit_config.metro_lines)}")
    print(f"  Fleet: {transit_config.total_buses} buses, "
          f"{transit_config.total_metro_trains} metro trains")
    print(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)


def print_optimization_report(final_state: dict):
    """Print the formatted transit optimization report."""
    plan = final_state.get("optimization_plan")
    if not plan:
        print("No optimization plan generated.")
        return

    status = plan.get("network_status", "unknown").upper()
    status_colors = {
        "CRITICAL": "[!!]", "DISRUPTED": "[!]",
        "DEGRADED": "[~]", "NORMAL": "[OK]", "ENHANCED": "[+]"
    }
    status_indicator = status_colors.get(status, "[ ]")

    print("\n" + "=" * 72)
    print(f"  TRANSIT OPERATIONS REPORT  {status_indicator} {status}")
    print("=" * 72)
    print(f"  Generated: {plan.get('generated_at', 'N/A')}")
    print(f"  Planning horizon: {plan.get('planning_horizon_hours')} hours")

    # KPI projections
    kpis = plan.get("kpi_projections", {})
    if kpis:
        print("\n--- PERFORMANCE METRICS ---")
        print(f"  On-time performance:   {kpis.get('on_time_performance_before', 0):.1f}% "
              f"(target: {kpis.get('on_time_performance_target', 85)}%)")
        print(f"  Overcrowded routes:    {kpis.get('overcrowded_routes_before', 0)} -> "
              f"{kpis.get('overcrowded_routes_after_projection', 0)} (projected)")
        print(f"  Vehicles redeployed:   {kpis.get('vehicles_redeployed', 0)}")
        print(f"  Schedule adjustments:  {kpis.get('total_adjustments', 0)}")
        print(f"  Alerts published:      {kpis.get('alerts_published', 0)}")

    # Schedule adjustments
    adjustments = plan.get("schedule_adjustments", [])
    if adjustments:
        print(f"\n--- SCHEDULE ADJUSTMENTS ({len(adjustments)}) ---")
        for adj in adjustments:
            headway_str = f" | headway={adj.get('new_headway_minutes')}min" if adj.get("new_headway_minutes") else ""
            vehicles_str = ""
            if adj.get("vehicles_added"):
                vehicles_str += f" +{adj.get('vehicles_added')}v"
            if adj.get("vehicles_removed"):
                vehicles_str += f" -{adj.get('vehicles_removed')}v"
            print(f"  {adj.get('route_id')} [{adj.get('route_type')}]: "
                  f"{adj.get('adjustment_type')}{headway_str}{vehicles_str} "
                  f"| [{adj.get('priority', 'normal').upper()}]")

    # Fleet deployments
    deployments = plan.get("fleet_deployments", [])
    if deployments:
        print(f"\n--- FLEET REDEPLOYMENTS ({len(deployments)}) ---")
        for dep in deployments:
            sources = ", ".join(dep.get("source_routes", []))
            print(f"  -> {dep.get('route_id')}: +{dep.get('vehicles_to_add')} vehicles "
                  f"from [{sources}]")

    # Service alerts
    alerts = plan.get("service_alerts", [])
    if alerts:
        print(f"\n--- PASSENGER ALERTS PUBLISHED ({len(alerts)}) ---")
        for alert in alerts:
            print(f"  [{alert.get('severity', 'info').upper()}] {alert.get('headline')}")
            print(f"    Routes: {alert.get('affected_routes')} | "
                  f"Channels: {alert.get('channels')}")

    # Executive summary
    print("\n--- OPERATIONS SUMMARY ---")
    print(plan.get("summary", "No summary available"))
    print("\n" + "=" * 72)


def save_report(final_state: dict, filename: str = None) -> str:
    """Save the optimization report to JSON."""
    if not filename:
        filename = f"transit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        json.dump(final_state, f, indent=2, default=str)

    print(f"\nReport saved to: {filename}")
    return filename


def run_optimization_cycle():
    """Execute a single transit optimization cycle."""
    print_header()

    if not transit_config.openai_api_key:
        print("\nERROR: OPENAI_API_KEY not set.")
        print("Create a .env file with: OPENAI_API_KEY=sk-your-key-here")
        sys.exit(1)

    print(get_graph_description())
    print("\nStarting transit optimization cycle...\n")
    print("-" * 72)

    graph = build_transit_graph()
    initial_state = AgentState().model_dump()

    config = {
        "configurable": {
            "thread_id": f"transit-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
    }

    final_state = None
    try:
        for step in graph.stream(initial_state, config=config):
            for node_name, state in step.items():
                print(f"  Completed node: [{node_name}]")
                final_state = state

    except Exception as e:
        print(f"\nERROR during execution: {e}")
        raise

    if final_state:
        print_optimization_report(final_state)
        save_report(final_state)
        return final_state
    else:
        print("No final state produced.")
        return None


if __name__ == "__main__":
    run_optimization_cycle()
