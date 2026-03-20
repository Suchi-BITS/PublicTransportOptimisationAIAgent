# data/simulation.py
# Transit simulation — date-seeded RNG for reproducible demo data.
#
# Production replacements:
#   get_fleet_positions()    -> GTFS-RT VehiclePositions feed (protobuf/JSON)
#   get_passenger_loads()    -> APC (Automatic Passenger Counter) API
#   get_incidents()          -> CAD (Computer-Aided Dispatch) system API
#   get_traffic_conditions() -> HERE Traffic API / TomTom Flow API
#   get_schedule()           -> GTFS static schedule + GTFS-RT TripUpdates

import random
from datetime import datetime, date
from typing import List, Dict

STOPS = {
    "R1-Red":    ["Central", "Eastgate", "Stadium", "Airport", "Northpark"],
    "R2-Blue":   ["Central", "Westside", "Hospital", "University", "Suburbs"],
    "R3-Green":  ["Downtown", "Market", "Riverside", "Industrial", "Eastgate"],
    "R4-Yellow": ["Airport", "BusinessPark", "Central", "Southend", "Port"],
    "R5-Express":["Central", "Airport", "University", "Northpark"],
}

VEHICLE_TYPES = {
    "R1-Red": "articulated_bus", "R2-Blue": "standard_bus",
    "R3-Green": "standard_bus",  "R4-Yellow": "minibus",
    "R5-Express": "express_coach",
}

CAPACITY = {
    "articulated_bus": 120, "standard_bus": 75,
    "minibus": 40,          "express_coach": 55,
}


def _rng(salt: int = 0) -> random.Random:
    return random.Random(int(date.today().strftime("%Y%m%d")) + salt)


def get_fleet_positions() -> List[Dict]:
    vehicles = []
    for i, route in enumerate(STOPS):
        r = _rng(i * 10)
        n_vehicles = r.randint(2, 4)
        stops = STOPS[route]
        for j in range(n_vehicles):
            r2 = _rng(i * 100 + j)
            stop_idx = r2.randint(0, len(stops) - 1)
            delay    = round(r2.gauss(0, 8), 1)
            cap      = CAPACITY[VEHICLE_TYPES[route]]
            load     = round(r2.uniform(0.3, 1.1) * cap)
            load     = min(load, cap)
            vehicles.append({
                "vehicle_id":          f"{route}-V{j+1:02d}",
                "route_id":            route,
                "current_stop":        stops[stop_idx],
                "next_stop":           stops[(stop_idx + 1) % len(stops)],
                "delay_minutes":       delay,
                "passenger_load":      load,
                "capacity":            cap,
                "occupancy_pct":       round(load / cap * 100, 1),
                "speed_kmh":           round(r2.uniform(12, 45), 1),
                "vehicle_type":        VEHICLE_TYPES[route],
                "operational_status":  r2.choice(["on_time", "on_time", "delayed", "ahead"]),
            })
    return vehicles


def get_incidents() -> List[Dict]:
    r = _rng(500)
    n = r.randint(0, 3)
    incident_types = ["road_closure", "breakdown", "signal_failure",
                      "passenger_incident", "traffic_congestion"]
    incidents = []
    for i in range(n):
        ri = _rng(500 + i)
        route = ri.choice(list(STOPS.keys()))
        incidents.append({
            "incident_id":    f"INC-{datetime.now().strftime('%Y%m%d')}-{i+1:03d}",
            "type":           ri.choice(incident_types),
            "route_affected": route,
            "stop_affected":  ri.choice(STOPS[route]),
            "severity":       ri.choice(["low", "medium", "high"]),
            "estimated_duration_minutes": ri.randint(5, 45),
            "reported_at":    datetime.now().isoformat(timespec="minutes"),
        })
    return incidents


def get_passenger_demand() -> List[Dict]:
    demand = []
    for i, route in enumerate(STOPS):
        r = _rng(700 + i)
        demand.append({
            "route_id":                route,
            "current_boardings_per_hr": r.randint(80, 450),
            "peak_stop":               r.choice(STOPS[route]),
            "avg_wait_minutes":        round(r.uniform(2, 18), 1),
            "demand_trend":            r.choice(["increasing", "stable", "decreasing"]),
        })
    return demand


def get_traffic_conditions() -> Dict:
    r = _rng(900)
    return {
        "overall_congestion":    r.choice(["low", "moderate", "high", "severe"]),
        "avg_speed_kmh":         round(r.uniform(12, 38), 1),
        "hotspots": [
            {"location": r.choice(["Central Junction", "Airport Road", "Market Street",
                                   "University Ave", "Industrial Bridge"]),
             "delay_minutes": round(r.uniform(3, 20), 1),
             "cause":         r.choice(["traffic", "road_works", "event"])}
            for _ in range(r.randint(1, 3))
        ],
    }
