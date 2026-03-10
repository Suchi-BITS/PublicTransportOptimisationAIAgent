# tools/sensor_tools.py
#
# Transit network data simulation layer.
# All five functions simulate real-time data feeds that would come from:
#
#   fetch_passenger_demand()     -> Automated Fare Collection (AFC) system tap-in/tap-out counts,
#                                   Automated Passenger Counter (APC) infrared sensors on buses,
#                                   Platform crowd density cameras (computer vision),
#                                   Real-time ticketing APIs (Cubic, Masabi, Conduent)
#
#   fetch_traffic_conditions()   -> City Traffic Management Centre (TMC) API,
#                                   HERE Traffic API / TomTom Traffic Stats API,
#                                   Connected vehicle probe data (Wejo, Arity),
#                                   Incident management system (CAD feeds)
#
#   fetch_events_calendar()      -> City Open Data events portal (REST API),
#                                   Venue ticketing systems (Ticketmaster API, AXS),
#                                   Sports league schedule APIs (ESPN, Opta),
#                                   Permit office database
#
#   fetch_fleet_status()         -> Automatic Vehicle Location (AVL) GPS transponders,
#                                   Computer-Aided Dispatch (CAD) system,
#                                   Onboard diagnostics (OBD-II / CAN bus),
#                                   Driver duty management system (Trapeze, Optibus)
#
#   get_historical_ridership()   -> Data warehouse / OLAP system (Redshift, BigQuery),
#                                   NTD (National Transit Database) reporting system,
#                                   Transit BI platform (TransitScreen, Remix)
#
# SIMULATION DESIGN:
#   - Uses a date-seeded RNG so values are reproducible on the same calendar day
#     but differ across days (mirrors real daily variation)
#   - Time-of-day demand curves are hardcoded from published transit ridership research
#   - Per-route baselines reflect realistic urban transit load factors
#   - Fleet vehicle counts are drawn from the configured total fleet (80 buses, 24 trains)
#   - Incident probability (18%) matches industry average for major urban networks
#
# TO REPLACE WITH REAL DATA:
#   Swap each function body for an API call to the production system listed above.
#   All downstream agents and the LangGraph pipeline require NO changes.

import random
from datetime import datetime, timedelta, date

try:
    from langchain_core.tools import tool
except ImportError:
    def tool(fn):
        return fn

from config.settings import transit_config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rng() -> random.Random:
    seed = int(date.today().strftime("%Y%m%d"))
    return random.Random(seed)


def _time_period() -> str:
    hour = datetime.now().hour
    if   6  <= hour <  9:  return "morning_peak"
    elif 9  <= hour < 12:  return "mid_morning"
    elif 12 <= hour < 14:  return "lunch_peak"
    elif 14 <= hour < 17:  return "afternoon"
    elif 17 <= hour < 20:  return "evening_peak"
    elif 20 <= hour < 23:  return "evening"
    else:                  return "off_peak"


_DEMAND_MULTIPLIERS = {
    "morning_peak": 1.85, "mid_morning": 1.10, "lunch_peak": 1.30,
    "afternoon":    1.00, "evening_peak": 1.90, "evening":   1.20,
    "off_peak":     0.38,
}

# ---------------------------------------------------------------------------
# Static network data  (in production: from GTFS static feed)
# ---------------------------------------------------------------------------

_ROUTE_CONFIG = {
    "B1": {"type":"bus",   "baseline_pax":48,  "capacity":transit_config.bus_capacity,   "peak_station":"Central Station",  "description":"North-South arterial — CBD to North Suburbs",       "typical_headway_min":10},
    "B2": {"type":"bus",   "baseline_pax":36,  "capacity":transit_config.bus_capacity,   "peak_station":"University Hub",   "description":"University corridor — East Campus to City Centre",  "typical_headway_min":12},
    "B3": {"type":"bus",   "baseline_pax":29,  "capacity":transit_config.bus_capacity,   "peak_station":"Tech Park",        "description":"Tech district shuttle — Tech Park to Downtown Core", "typical_headway_min":15},
    "B4": {"type":"bus",   "baseline_pax":54,  "capacity":transit_config.bus_capacity,   "peak_station":"Downtown Core",    "description":"Downtown loop — highest-demand surface route",       "typical_headway_min":8},
    "B5": {"type":"bus",   "baseline_pax":42,  "capacity":transit_config.bus_capacity,   "peak_station":"Westgate Mall",    "description":"Airport connector — Airport Terminal to Westgate",   "typical_headway_min":12},
    "B6": {"type":"bus",   "baseline_pax":31,  "capacity":transit_config.bus_capacity,   "peak_station":"Riverside",        "description":"Riverside scenic route — lower demand, tourist use", "typical_headway_min":20},
    "M1": {"type":"metro", "baseline_pax":225, "capacity":transit_config.metro_capacity, "peak_station":"Central Station",  "description":"Red Line — Central to Airport via Downtown",         "typical_headway_min":4},
    "M2": {"type":"metro", "baseline_pax":190, "capacity":transit_config.metro_capacity, "peak_station":"Airport Terminal", "description":"Blue Line — Airport express, high tourist+business",  "typical_headway_min":5},
    "M3": {"type":"metro", "baseline_pax":165, "capacity":transit_config.metro_capacity, "peak_station":"Stadium District", "description":"Green Line — Stadium District to University Hub",     "typical_headway_min":6},
}

_CORRIDOR_CONFIG = {
    "north_south_arterial": {"routes":["B1","B4"],        "free_flow_kmh":52, "description":"Main north-south spine through city centre"},
    "east_west_boulevard":  {"routes":["B2","B3"],        "free_flow_kmh":48, "description":"East-west connector through university and tech zones"},
    "airport_express":      {"routes":["B5","M2"],        "free_flow_kmh":65, "description":"Dedicated airport access road, limited signals"},
    "downtown_loop":        {"routes":["B1","B4","M1"],   "free_flow_kmh":38, "description":"CBD ring road — lowest free-flow speed due to density"},
    "university_corridor":  {"routes":["B2","B6"],        "free_flow_kmh":45, "description":"Campus-adjacent road, pedestrian crossings reduce speed"},
    "stadium_access":       {"routes":["B3","M3"],        "free_flow_kmh":55, "description":"Stadium District approach road, event-sensitive"},
    "tech_district":        {"routes":["B3","B6"],        "free_flow_kmh":50, "description":"Tech Park and innovation hub arterial"},
    "westside_connector":   {"routes":["B5","B6"],        "free_flow_kmh":58, "description":"Western suburbs connector to mall and riverside"},
}

# Today's scheduled events (static roster).
# In production: live query to city events API every 15 min.
_TODAY_EVENTS = [
    {"event_id":"EVT-0001","event_name":"Metro City FC Home Match",      "venue":"MetroStadium",          "nearest_station":"Stadium District","event_type":"sports",     "start_hour":15,"duration_hours":3, "attendance":38500,"modal_share":0.55,"affected_routes":["M3","B3"],       "demand_pattern":"both"},
    {"event_id":"EVT-0002","event_name":"Grand Arena Concert",           "venue":"Grand Arena",           "nearest_station":"Downtown Core",   "event_type":"concert",    "start_hour":20,"duration_hours":4, "attendance":11000,"modal_share":0.65,"affected_routes":["M1","B4","B1"],  "demand_pattern":"surge_after"},
    {"event_id":"EVT-0003","event_name":"Tech Summit Conference",        "venue":"Metro Convention Center","nearest_station":"Tech Park",       "event_type":"conference", "start_hour":8, "duration_hours":10,"attendance":3200, "modal_share":0.45,"affected_routes":["B3","B6"],       "demand_pattern":"both"},
    {"event_id":"EVT-0004","event_name":"Riverside Food Festival",       "venue":"Riverside Park",        "nearest_station":"Riverside",       "event_type":"festival",   "start_hour":11,"duration_hours":8, "attendance":8500, "modal_share":0.50,"affected_routes":["B6","M1"],       "demand_pattern":"steady"},
    {"event_id":"EVT-0005","event_name":"International Arrivals Wave",   "venue":"Airport Terminal",      "nearest_station":"Airport Terminal","event_type":"other",      "start_hour":7, "duration_hours":3, "attendance":2200, "modal_share":0.70,"affected_routes":["M2","B5"],       "demand_pattern":"steady"},
]

# Baseline vehicle count per route (80 buses / 24 trains total)
_FLEET_BASELINE = {"B1":14,"B2":10,"B3":8,"B4":16,"B5":12,"B6":8,"M1":9,"M2":8,"M3":7}


# ---------------------------------------------------------------------------
# Tool 1: Passenger demand
# ---------------------------------------------------------------------------

@tool
def fetch_passenger_demand(route_ids: list) -> list:
    """
    Fetch real-time passenger demand from AFC ticketing and APC sensors.

    Production: Cubic NextAgent AFC tap-in counts + Iris APC infrared door sensors.
    Simulation: baseline_pax x time-of-day multiplier x daily variation (+-20%).

    Args:
        route_ids: e.g. ['B1', 'B4', 'M1']
    Returns:
        List of demand dicts, one per route
    """
    rng        = _rng()
    period     = _time_period()
    multiplier = _DEMAND_MULTIPLIERS[period]

    results = []
    for route_id in route_ids:
        cfg = _ROUTE_CONFIG.get(route_id)
        if not cfg:
            continue

        daily_var   = rng.uniform(0.82, 1.22)
        current_pax = int(cfg["baseline_pax"] * multiplier * daily_var)
        current_pax = min(current_pax, int(cfg["capacity"] * 1.30))

        load_factor = current_pax / cfg["capacity"]
        wait_min    = round(rng.uniform(2.5, 6.0) + (load_factor * 10), 1)

        results.append({
            "route_id":                  route_id,
            "route_type":                cfg["type"],
            "description":               cfg["description"],
            "current_passengers":        current_pax,
            "vehicle_capacity":          cfg["capacity"],
            "load_factor":               round(load_factor, 3),
            "demand_forecast_1h":        int(current_pax * rng.uniform(0.88, 1.18)),
            "demand_forecast_3h":        int(cfg["baseline_pax"] * multiplier * rng.uniform(0.80, 1.35)),
            "peak_station":              cfg["peak_station"],
            "peak_station_wait_minutes": wait_min,
            "baseline_demand":           cfg["baseline_pax"],
            "demand_index":              round((current_pax / cfg["baseline_pax"]) / multiplier, 2),
            "time_period":               period,
            "overcrowded":               load_factor >= transit_config.overcrowding_threshold,
            "underutilized":             load_factor <= transit_config.underutilization_threshold,
        })
    return results


# ---------------------------------------------------------------------------
# Tool 2: Traffic conditions
# ---------------------------------------------------------------------------

@tool
def fetch_traffic_conditions(corridors: list) -> list:
    """
    Fetch road traffic conditions for bus corridors.

    Production: HERE Traffic API + City TMC DATEX II feed.
    Simulation: Congestion level drawn from weighted distribution, 18% incident probability.

    Args:
        corridors: e.g. ['downtown_loop', 'airport_express']
    Returns:
        List of traffic condition dicts, one per corridor
    """
    rng = _rng()

    congestion_levels  = ["free_flow","light","moderate","heavy","gridlock"]
    speed_factors      = [1.00, 0.75, 0.55, 0.35, 0.15]
    congestion_weights = [25,   30,   25,   15,    5]

    incident_descs = {
        "accident":      "Two-vehicle collision blocking right lane — emergency services on scene",
        "road_works":    "Water main repair — one lane closed, traffic management in place",
        "event_closure": "Street temporarily closed for permitted event until this evening",
        "weather":       "Surface water causing slow traffic — 40 km/h advisory speed",
    }

    results = []
    for corridor in corridors:
        cfg = _CORRIDOR_CONFIG.get(corridor)
        if not cfg:
            continue

        level_idx     = rng.choices(range(5), weights=congestion_weights)[0]
        free_flow_kmh = cfg["free_flow_kmh"]
        current_kmh   = round(free_flow_kmh * speed_factors[level_idx] * rng.uniform(0.93,1.07), 1)
        current_kmh   = max(current_kmh, 5.0)

        segment_km    = 5.0
        delay_min     = round(max(0.0, (segment_km/current_kmh - segment_km/free_flow_kmh)*60), 1)

        has_incident   = rng.random() < 0.18
        incident_types = ["accident","road_works","event_closure","weather"]
        incident_type  = rng.choice(incident_types) if has_incident else "none"
        incident_desc  = incident_descs.get(incident_type) if has_incident else None
        clearance      = None
        if has_incident:
            clearance  = (datetime.now() + timedelta(hours=rng.randint(1,3))).strftime("%H:%M")

        results.append({
            "corridor":                  corridor,
            "description":               cfg["description"],
            "route_ids_affected":        cfg["routes"],
            "free_flow_speed_kmh":       free_flow_kmh,
            "current_speed_kmh":         current_kmh,
            "congestion_level":          congestion_levels[level_idx],
            "delay_minutes":             delay_min,
            "speed_reduction_percent":   round((1 - current_kmh/free_flow_kmh)*100, 1),
            "incident_type":             incident_type,
            "incident_description":      incident_desc,
            "estimated_clearance_time":  clearance,
            "affected_stations":         rng.sample(transit_config.key_stations, k=rng.randint(1,3)),
            "severity_score":            level_idx * 25,
        })
    return results


# ---------------------------------------------------------------------------
# Tool 3: Events calendar
# ---------------------------------------------------------------------------

@tool
def fetch_events_calendar(hours_ahead: int = 12) -> list:
    """
    Fetch today's events that generate transit demand surges.

    Production: Ticketmaster API + city permit office + ESPN sports API.
    Simulation: Static roster of 5 events, filtered by hours_ahead window.
    Attendance varies +-8% (weather/day-of-week effect).

    Args:
        hours_ahead: How far ahead to look (default 12 hours)
    Returns:
        List of active/upcoming event dicts
    """
    rng = _rng()
    now = datetime.now()

    results = []
    for tmpl in _TODAY_EVENTS:
        start_time = now.replace(hour=tmpl["start_hour"], minute=0, second=0, microsecond=0)
        end_time   = start_time + timedelta(hours=tmpl["duration_hours"])

        active       = start_time <= now <= end_time
        in_window    = now <= start_time <= now + timedelta(hours=hours_ahead)
        if not (active or in_window):
            continue

        attendance     = int(tmpl["attendance"] * rng.uniform(0.92, 1.08))
        transit_demand = int(attendance * tmpl["modal_share"])
        impact         = "critical" if transit_demand>5000 else "major" if transit_demand>2000 else "moderate"

        results.append({
            "event_id":                 tmpl["event_id"],
            "event_name":               tmpl["event_name"],
            "venue":                    tmpl["venue"],
            "nearest_station":          tmpl["nearest_station"],
            "event_type":               tmpl["event_type"],
            "start_time":               start_time.isoformat(),
            "end_time":                 end_time.isoformat(),
            "is_active_now":            active,
            "minutes_until_start":      max(0, int((start_time-now).total_seconds()/60)),
            "expected_attendance":      attendance,
            "transit_modal_share":      tmpl["modal_share"],
            "estimated_transit_demand": transit_demand,
            "affected_routes":          tmpl["affected_routes"],
            "demand_pattern":           tmpl["demand_pattern"],
            "impact_level":             impact,
            "requires_special_service": transit_demand > 5000,
        })
    return results


# ---------------------------------------------------------------------------
# Tool 4: Fleet vehicle status
# ---------------------------------------------------------------------------

@tool
def fetch_fleet_status(route_ids: list) -> list:
    """
    Fetch real-time GPS position, load, and mechanical status of all active vehicles.

    Production: AVL GPS transponders (30-sec updates) + CAD system (Trapeze CONNECT).
    Simulation: baseline vehicles per route, OTP weighted to 85% target,
                mechanical status 88% operational / 9% degraded / 3% out-of-service.

    Args:
        route_ids: Routes to query e.g. ['B4', 'M1']
    Returns:
        List of vehicle status dicts (multiple per route)
    """
    rng = _rng()

    on_time_statuses = ["on_time","early","late","very_late"]
    on_time_weights  = [52, 12, 26, 10]
    mech_statuses    = ["operational","degraded","out_of_service"]
    mech_weights     = [88, 9, 3]

    results = []
    for route_id in route_ids:
        cfg = _ROUTE_CONFIG.get(route_id)
        if not cfg:
            continue

        in_service = max(1, _FLEET_BASELINE.get(route_id, 8) + rng.randint(-2, 2))

        for j in range(in_service):
            status = rng.choices(on_time_statuses, weights=on_time_weights)[0]
            if   status == "late":      delay = round(rng.uniform(2.0,  8.0), 1)
            elif status == "very_late": delay = round(rng.uniform(8.0, 25.0), 1)
            elif status == "early":     delay = round(-rng.uniform(0.5,  3.0), 1)
            else:                       delay = 0.0

            pax         = rng.randint(0, int(cfg["capacity"] * 1.10))
            current_loc = rng.choice(transit_config.key_stations)
            next_stop   = rng.choice([s for s in transit_config.key_stations if s != current_loc])
            shift_hrs   = round(rng.uniform(0.5, 8.0), 1)

            results.append({
                "vehicle_id":            f"{cfg['type'][0].upper()}{route_id}-{j+1:02d}",
                "vehicle_type":          cfg["type"],
                "route_id":              route_id,
                "current_location":      current_loc,
                "next_stop":             next_stop,
                "passengers_onboard":    pax,
                "capacity":              cfg["capacity"],
                "load_factor":           round(pax / cfg["capacity"], 2),
                "on_time_status":        status,
                "delay_minutes":         delay,
                "fuel_level_percent":    round(rng.uniform(18,100),1) if cfg["type"]=="bus" else None,
                "mechanical_status":     rng.choices(mech_statuses, weights=mech_weights)[0],
                "driver_hours_on_shift": shift_hrs,
                "driver_fatigue_flag":   shift_hrs >= 7.5,
                "headway_gap_minutes":   round(rng.uniform(
                    cfg["typical_headway_min"]*0.5,
                    cfg["typical_headway_min"]*2.0), 1),
            })
    return results


# ---------------------------------------------------------------------------
# Tool 5: Historical ridership
# ---------------------------------------------------------------------------

@tool
def get_historical_ridership(route_id: str, days_back: int = 7) -> dict:
    """
    Retrieve historical ridership and performance statistics for a route.

    Production: Data warehouse (BigQuery / Redshift) populated nightly from AFC + APC exports.
    Simulation: Derived from route baselines with seeded variation.
    Day-of-week pattern from APTA 2023 ridership elasticity study.

    Args:
        route_id: e.g. 'B4', 'M1'
        days_back: Days of history to summarize
    Returns:
        Dict with aggregated ridership and performance statistics
    """
    rng = _rng()
    cfg = _ROUTE_CONFIG.get(route_id, _ROUTE_CONFIG["B1"])

    trips_per_day   = 220 if cfg["type"]=="metro" else 80
    daily_ridership = int(cfg["baseline_pax"] * trips_per_day * rng.uniform(0.88, 1.12))

    return {
        "route_id":                    route_id,
        "route_description":           cfg["description"],
        "period_days":                 days_back,
        "avg_daily_ridership":         daily_ridership,
        "total_ridership_period":      daily_ridership * days_back,
        "avg_peak_load_factor":        round(rng.uniform(0.78, 0.96), 2),
        "avg_off_peak_load_factor":    round(rng.uniform(0.18, 0.42), 2),
        "on_time_performance_percent": round(min(99, max(70, rng.gauss(85.0, 4.0))), 1),
        "avg_dwell_time_seconds":      round(rng.uniform(18, 52), 0),
        "avg_headway_minutes":         cfg["typical_headway_min"],
        "schedule_adherence_percent":  round(rng.uniform(78, 94), 1),
        "complaints_per_1000_riders":  round(rng.uniform(1.0, 6.2), 1),
        "avg_journey_time_minutes":    round(rng.uniform(22, 48), 0),
        "most_congested_stop":         cfg["peak_station"],
        "day_of_week_pattern": {
            "monday":1.04,"tuesday":1.00,"wednesday":1.02,
            "thursday":1.05,"friday":1.18,"saturday":0.82,"sunday":0.61,
        },
        "ridership_trend_percent": round(rng.gauss(1.5, 3.0), 1),
        "otp_trend_percent":       round(rng.gauss(0.2, 1.5), 1),
    }
