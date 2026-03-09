# tools/sensor_tools.py
# Simulated transit data collection tools
# In production: connect to GTFS-RT feeds, traffic APIs, ticketing systems, event APIs

import random
from datetime import datetime, timedelta
from langchain_core.tools import tool
from data.models import (
    PassengerDemandData, TrafficConditionData,
    EventData, VehicleStatus
)
from config.settings import transit_config


def _current_time_period() -> str:
    """Determine current time period for demand modeling."""
    hour = datetime.now().hour
    if 6 <= hour < 9:
        return "morning_peak"
    elif 9 <= hour < 12:
        return "mid_morning"
    elif 12 <= hour < 14:
        return "lunch_peak"
    elif 14 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 20:
        return "evening_peak"
    elif 20 <= hour < 23:
        return "evening"
    else:
        return "off_peak"


def _demand_multiplier(time_period: str) -> float:
    """Get demand multiplier based on time period."""
    multipliers = {
        "morning_peak": 1.8,
        "mid_morning": 1.1,
        "lunch_peak": 1.3,
        "afternoon": 1.0,
        "evening_peak": 1.9,
        "evening": 1.2,
        "off_peak": 0.4
    }
    return multipliers.get(time_period, 1.0)


@tool
def fetch_passenger_demand(route_ids: list[str]) -> list[dict]:
    """
    Fetch real-time passenger demand data from ticketing systems and
    automated passenger counters (APC) for specified routes.

    In production connects to: AFC (Automated Fare Collection) systems,
    APC sensors on vehicles, platform crowd sensors, and demand forecasting models.

    Args:
        route_ids: List of route identifiers to fetch demand for

    Returns:
        List of demand data records per route
    """
    time_period = _current_time_period()
    base_multiplier = _demand_multiplier(time_period)

    route_config = {
        "B1": {"type": "bus", "baseline": 45, "capacity": transit_config.bus_capacity,
               "peak_station": "Central Station"},
        "B2": {"type": "bus", "baseline": 35, "capacity": transit_config.bus_capacity,
               "peak_station": "University Hub"},
        "B3": {"type": "bus", "baseline": 28, "capacity": transit_config.bus_capacity,
               "peak_station": "Tech Park"},
        "B4": {"type": "bus", "baseline": 52, "capacity": transit_config.bus_capacity,
               "peak_station": "Downtown Core"},
        "B5": {"type": "bus", "baseline": 40, "capacity": transit_config.bus_capacity,
               "peak_station": "Westgate Mall"},
        "B6": {"type": "bus", "baseline": 30, "capacity": transit_config.bus_capacity,
               "peak_station": "Riverside"},
        "M1": {"type": "metro", "baseline": 220, "capacity": transit_config.metro_capacity,
               "peak_station": "Central Station"},
        "M2": {"type": "metro", "baseline": 185, "capacity": transit_config.metro_capacity,
               "peak_station": "Airport Terminal"},
        "M3": {"type": "metro", "baseline": 160, "capacity": transit_config.metro_capacity,
               "peak_station": "Stadium District"},
    }

    results = []
    for route_id in route_ids:
        cfg = route_config.get(route_id, {
            "type": "bus", "baseline": 30, "capacity": 60, "peak_station": "Downtown Core"
        })

        baseline = cfg["baseline"]
        variation = random.uniform(0.75, 1.35)
        current = int(baseline * base_multiplier * variation)
        current = min(current, int(cfg["capacity"] * 1.3))  # can exceed capacity (standing)

        demand = PassengerDemandData(
            route_id=route_id,
            route_type=cfg["type"],
            current_passengers=current,
            vehicle_capacity=cfg["capacity"],
            load_factor=round(current / cfg["capacity"], 3),
            demand_forecast_1h=int(current * random.uniform(0.85, 1.25)),
            demand_forecast_3h=int(baseline * base_multiplier * random.uniform(0.8, 1.4)),
            peak_station=cfg["peak_station"],
            peak_station_wait_minutes=round(random.uniform(2, 18), 1),
            baseline_demand=baseline,
            demand_index=round((current / baseline) / base_multiplier, 2)
        )
        results.append(demand.model_dump(mode="json"))

    return results


@tool
def fetch_traffic_conditions(corridors: list[str]) -> list[dict]:
    """
    Fetch current traffic and road condition data for transit corridors.

    In production connects to: city traffic management systems, HERE/TomTom APIs,
    connected vehicle data, incident management systems.

    Args:
        corridors: List of road corridor or transit segment identifiers

    Returns:
        List of traffic condition records per corridor
    """
    corridor_routes = {
        "north_south_arterial": ["B1", "B4"],
        "east_west_boulevard": ["B2", "B3"],
        "airport_express": ["B5", "M2"],
        "downtown_loop": ["B1", "B4", "M1"],
        "university_corridor": ["B2", "B6"],
        "stadium_access": ["B3", "M3"],
        "tech_district": ["B3", "B6"],
        "westside_connector": ["B5", "B6"]
    }

    congestion_levels = ["free_flow", "light", "moderate", "heavy", "gridlock"]
    incident_types = ["accident", "road_works", "event_closure", "weather", "none"]

    results = []
    for corridor in corridors:
        affected_routes = corridor_routes.get(corridor, ["B1"])
        free_flow = random.uniform(45, 65)
        congestion_weight = random.choices(
            [0, 1, 2, 3, 4],
            weights=[25, 30, 25, 15, 5]
        )[0]
        congestion_label = congestion_levels[congestion_weight]
        speed_reduction = [1.0, 0.75, 0.55, 0.35, 0.15][congestion_weight]
        current_speed = round(free_flow * speed_reduction, 1)
        delay = round((free_flow / max(current_speed, 5) - 1) * random.uniform(8, 15), 1)

        has_incident = random.random() < 0.20
        incident = random.choice(incident_types[:-1]) if has_incident else "none"
        incident_desc = None
        clearance = None

        if has_incident:
            descs = {
                "accident": "Two-vehicle collision blocking right lane",
                "road_works": "Water main repair, lane closure in effect",
                "event_closure": "Street closed for permitted event",
                "weather": "Flooding causing partial road closure"
            }
            incident_desc = descs.get(incident)
            clearance = (datetime.now() + timedelta(hours=random.randint(1, 4))).strftime("%H:%M")

        cond = TrafficConditionData(
            corridor=corridor,
            route_ids_affected=affected_routes,
            current_speed_kmh=current_speed,
            free_flow_speed_kmh=free_flow,
            congestion_level=congestion_label,
            delay_minutes=max(0.0, delay),
            incident_type=incident,
            incident_description=incident_desc,
            estimated_clearance_time=clearance,
            affected_stations=random.sample(transit_config.key_stations, k=random.randint(1, 3))
        )
        results.append(cond.model_dump(mode="json"))

    return results


@tool
def fetch_events_calendar(hours_ahead: int = 12) -> list[dict]:
    """
    Fetch scheduled events that will generate transit demand surges.

    In production connects to: city events API, venue ticketing systems,
    permit office database, sports league APIs, concert venue feeds.

    Args:
        hours_ahead: How many hours ahead to look for events

    Returns:
        List of events with estimated transit demand impact
    """
    now = datetime.now()
    event_templates = [
        {
            "event_name": "Metro City FC Home Match",
            "venue": "MetroStadium",
            "nearest_station": "Stadium District",
            "event_type": "sports",
            "duration_hours": 3,
            "attendance": random.randint(28000, 45000),
            "modal_share": 0.55,
            "affected_routes": ["M3", "B3"],
            "pattern": "both"
        },
        {
            "event_name": "Grand Arena Concert",
            "venue": "Grand Arena",
            "nearest_station": "Downtown Core",
            "event_type": "concert",
            "duration_hours": 4,
            "attendance": random.randint(8000, 15000),
            "modal_share": 0.65,
            "affected_routes": ["M1", "B4", "B1"],
            "pattern": "surge_after"
        },
        {
            "event_name": "Tech Summit Conference",
            "venue": "Metro Convention Center",
            "nearest_station": "Tech Park",
            "event_type": "conference",
            "duration_hours": 8,
            "attendance": random.randint(2000, 5000),
            "modal_share": 0.45,
            "affected_routes": ["B3", "B6"],
            "pattern": "both"
        },
        {
            "event_name": "Riverside Food Festival",
            "venue": "Riverside Park",
            "nearest_station": "Riverside",
            "event_type": "festival",
            "duration_hours": 6,
            "attendance": random.randint(5000, 12000),
            "modal_share": 0.50,
            "affected_routes": ["B6", "M1"],
            "pattern": "steady"
        },
        {
            "event_name": "Airport International Arrivals Wave",
            "venue": "Airport Terminal",
            "nearest_station": "Airport Terminal",
            "event_type": "other",
            "duration_hours": 2,
            "attendance": random.randint(1500, 3000),
            "modal_share": 0.70,
            "affected_routes": ["M2", "B5"],
            "pattern": "steady"
        }
    ]

    # Randomly select 2-4 active events
    active_events = random.sample(event_templates, k=random.randint(2, 4))
    results = []

    for i, tmpl in enumerate(active_events):
        start_offset = random.randint(0, max(1, hours_ahead - 2))
        start_time = now + timedelta(hours=start_offset)
        end_time = start_time + timedelta(hours=tmpl["duration_hours"])

        if start_time > now + timedelta(hours=hours_ahead):
            continue

        attendance = tmpl["attendance"]
        modal_share = tmpl["modal_share"]
        transit_demand = int(attendance * modal_share)

        event = EventData(
            event_id=f"EVT-{i+1:04d}",
            event_name=tmpl["event_name"],
            venue=tmpl["venue"],
            nearest_station=tmpl["nearest_station"],
            event_type=tmpl["event_type"],
            start_time=start_time,
            end_time=end_time,
            expected_attendance=attendance,
            transit_modal_share=modal_share,
            estimated_transit_demand=transit_demand,
            affected_routes=tmpl["affected_routes"],
            demand_pattern=tmpl["pattern"],
            requires_special_service=transit_demand > 5000
        )
        results.append(event.model_dump(mode="json"))

    return results


@tool
def fetch_fleet_status(route_ids: list[str]) -> list[dict]:
    """
    Fetch real-time status of all vehicles currently in service.

    In production connects to: AVL (Automatic Vehicle Location) systems,
    CAD (Computer-Aided Dispatch), onboard diagnostics, driver terminals.

    Args:
        route_ids: Routes to fetch vehicle status for

    Returns:
        List of individual vehicle status records
    """
    on_time_weights = [50, 20, 20, 10]  # on_time, early, late, very_late

    results = []
    for route_id in route_ids:
        v_type = "metro" if route_id.startswith("M") else "bus"
        capacity = transit_config.metro_capacity if v_type == "metro" else transit_config.bus_capacity
        num_vehicles = random.randint(2, 5) if v_type == "metro" else random.randint(3, 8)

        stations = random.sample(transit_config.key_stations, k=min(4, len(transit_config.key_stations)))

        for j in range(num_vehicles):
            on_time = random.choices(
                ["on_time", "early", "late", "very_late"],
                weights=on_time_weights
            )[0]
            delay = 0.0
            if on_time == "late":
                delay = random.uniform(2, 8)
            elif on_time == "very_late":
                delay = random.uniform(8, 25)
            elif on_time == "early":
                delay = -random.uniform(0.5, 3)

            pax = random.randint(0, int(capacity * 1.1))

            vehicle = VehicleStatus(
                vehicle_id=f"{v_type.upper()[0]}{route_id}-{j+1:02d}",
                vehicle_type=v_type,
                route_id=route_id,
                current_location=random.choice(stations),
                next_stop=random.choice(stations),
                passengers_onboard=pax,
                capacity=capacity,
                on_time_status=on_time,
                delay_minutes=round(delay, 1),
                fuel_level_percent=round(random.uniform(25, 100), 1) if v_type == "bus" else None,
                mechanical_status=random.choices(
                    ["operational", "degraded", "out_of_service"],
                    weights=[88, 9, 3]
                )[0],
                driver_hours_on_shift=round(random.uniform(0.5, 7.5), 1)
            )
            results.append(vehicle.model_dump(mode="json"))

    return results


@tool
def get_historical_ridership(route_id: str, days_back: int = 7) -> dict:
    """
    Retrieve historical ridership patterns for a route.
    Used for trend analysis, anomaly detection, and demand forecasting validation.

    Args:
        route_id: Route to query history for
        days_back: Number of days of history to retrieve

    Returns:
        Dictionary with ridership stats and patterns
    """
    v_type = "metro" if route_id.startswith("M") else "bus"
    baseline_pax = random.randint(180, 350) if v_type == "metro" else random.randint(40, 80)

    return {
        "route_id": route_id,
        "period_days": days_back,
        "avg_daily_ridership": baseline_pax * random.randint(14, 22),
        "avg_peak_load_factor": round(random.uniform(0.75, 0.95), 2),
        "avg_off_peak_load_factor": round(random.uniform(0.20, 0.45), 2),
        "on_time_performance_percent": round(random.uniform(78, 92), 1),
        "avg_dwell_time_seconds": round(random.uniform(20, 55), 0),
        "complaints_per_1000_riders": round(random.uniform(1.2, 6.5), 1),
        "most_congested_stop": random.choice(transit_config.key_stations),
        "day_of_week_pattern": {
            "monday": 1.05,
            "tuesday": 1.0,
            "wednesday": 1.02,
            "thursday": 1.03,
            "friday": 1.15,
            "saturday": 0.85,
            "sunday": 0.65
        }
    }
