# tools/action_tools.py
# Tools for executing transit schedule adjustments and dispatching actions

from datetime import datetime
from langchain_core.tools import tool

# In-memory action log (production: database + CAD system integration)
_action_log: list[dict] = []


@tool
def adjust_route_frequency(
    route_id: str,
    route_type: str,
    adjustment_type: str,
    new_headway_minutes: int,
    direction: str,
    vehicles_added: int,
    vehicles_removed: int,
    priority: str,
    reason: str
) -> dict:
    """
    Modify the frequency or headway of a transit route.

    Args:
        route_id: Route identifier (e.g., 'B1', 'M2')
        route_type: 'bus' or 'metro'
        adjustment_type: 'increase_frequency', 'decrease_frequency', 'deploy_extra_vehicle', 'recall_vehicle'
        new_headway_minutes: New target gap between vehicles in minutes
        direction: 'inbound', 'outbound', or 'both'
        vehicles_added: Number of additional vehicles being deployed
        vehicles_removed: Number of vehicles being withdrawn
        priority: 'critical', 'high', 'normal', or 'low'
        reason: Explanation for this adjustment

    Returns:
        Confirmation with action ID
    """
    action_record = {
        "action_id": f"FREQ-{datetime.now().strftime('%Y%m%d%H%M%S')}-{route_id}",
        "type": "frequency_adjustment",
        "route_id": route_id,
        "route_type": route_type,
        "adjustment_type": adjustment_type,
        "new_headway_minutes": new_headway_minutes,
        "direction": direction,
        "vehicles_added": vehicles_added,
        "vehicles_removed": vehicles_removed,
        "priority": priority,
        "reason": reason,
        "status": "dispatched",
        "created_at": datetime.now().isoformat()
    }
    _action_log.append(action_record)

    print(f"  [DISPATCH] {route_id}: {adjustment_type} -> {new_headway_minutes} min headway "
          f"(+{vehicles_added}/-{vehicles_removed} vehicles) [{priority}]")

    return {
        "success": True,
        "action_id": action_record["action_id"],
        "message": f"Route {route_id} frequency adjusted: new headway {new_headway_minutes} min, "
                   f"{vehicles_added} vehicles added, {vehicles_removed} removed",
        "priority": priority
    }


@tool
def deploy_extra_service(
    route_id: str,
    route_type: str,
    service_type: str,
    num_vehicles: int,
    origin_station: str,
    destination_station: str,
    start_time: str,
    end_time: str,
    reason: str
) -> dict:
    """
    Deploy extra or special service trips for demand surges or events.

    Args:
        route_id: Route to add extra service to
        route_type: 'bus' or 'metro'
        service_type: 'express_service', 'short_turn', 'extend_route', or 'special_event_shuttle'
        num_vehicles: Number of extra vehicles to deploy
        origin_station: Start terminal for the extra service
        destination_station: End terminal for the extra service
        start_time: When extra service begins (ISO format or HH:MM)
        end_time: When extra service ends (ISO format or HH:MM)
        reason: Justification for extra service

    Returns:
        Deployment confirmation
    """
    action_record = {
        "action_id": f"XSVC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{route_id}",
        "type": "extra_service",
        "route_id": route_id,
        "route_type": route_type,
        "service_type": service_type,
        "num_vehicles": num_vehicles,
        "origin_station": origin_station,
        "destination_station": destination_station,
        "start_time": start_time,
        "end_time": end_time,
        "reason": reason,
        "status": "dispatched",
        "created_at": datetime.now().isoformat()
    }
    _action_log.append(action_record)

    print(f"  [DISPATCH] Extra service on {route_id}: {num_vehicles} {service_type} vehicles "
          f"{origin_station} -> {destination_station} ({start_time} - {end_time})")

    return {
        "success": True,
        "action_id": action_record["action_id"],
        "message": f"Extra {service_type} deployed on {route_id}: {num_vehicles} vehicles, "
                   f"{origin_station} to {destination_station}",
        "vehicles_deployed": num_vehicles
    }


@tool
def reallocate_fleet(
    from_route_id: str,
    to_route_id: str,
    num_vehicles: int,
    reason: str,
    effective_time: str
) -> dict:
    """
    Transfer vehicles from an underutilized route to an overcrowded one.

    Args:
        from_route_id: Route to withdraw vehicles from
        to_route_id: Route to assign withdrawn vehicles to
        num_vehicles: Number of vehicles to transfer
        reason: Justification for the reallocation
        effective_time: When the reallocation takes effect (HH:MM or ISO format)

    Returns:
        Reallocation confirmation
    """
    action_record = {
        "action_id": f"REALLOC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "type": "fleet_reallocation",
        "from_route": from_route_id,
        "to_route": to_route_id,
        "num_vehicles": num_vehicles,
        "reason": reason,
        "effective_time": effective_time,
        "status": "dispatched",
        "created_at": datetime.now().isoformat()
    }
    _action_log.append(action_record)

    print(f"  [DISPATCH] Fleet reallocation: {num_vehicles} vehicles from {from_route_id} "
          f"-> {to_route_id} at {effective_time}")

    return {
        "success": True,
        "action_id": action_record["action_id"],
        "message": f"{num_vehicles} vehicles transferred from {from_route_id} to {to_route_id}",
        "effective_time": effective_time
    }


@tool
def issue_service_alert(
    severity: str,
    alert_type: str,
    affected_routes: list[str],
    affected_stations: list[str],
    headline: str,
    body: str,
    alternative_routes: list[str],
    channels: list[str]
) -> dict:
    """
    Publish a passenger-facing service alert across all communication channels.

    Args:
        severity: 'critical', 'major', 'minor', or 'info'
        alert_type: 'delay', 'disruption', 'route_change', 'extra_service', 'cancellation', 'event_service', 'general'
        affected_routes: List of route IDs affected
        affected_stations: List of station names affected
        headline: Short alert headline (max 80 chars)
        body: Full alert description for passengers
        alternative_routes: Suggested alternative routes
        channels: Distribution channels - 'app', 'website', 'station_displays', 'social_media', 'sms'

    Returns:
        Alert publication confirmation
    """
    action_record = {
        "action_id": f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "type": "service_alert",
        "severity": severity,
        "alert_type": alert_type,
        "affected_routes": affected_routes,
        "affected_stations": affected_stations,
        "headline": headline,
        "body": body,
        "alternative_routes": alternative_routes,
        "channels": channels,
        "published_at": datetime.now().isoformat()
    }
    _action_log.append(action_record)

    print(f"  [ALERT - {severity.upper()}] {headline}")

    return {
        "success": True,
        "action_id": action_record["action_id"],
        "headline": headline,
        "channels_published": channels,
        "routes_affected": len(affected_routes)
    }


@tool
def hold_vehicle_at_station(
    vehicle_id: str,
    route_id: str,
    station: str,
    hold_minutes: int,
    reason: str
) -> dict:
    """
    Instruct a specific vehicle to hold at a station to regulate headways or await passengers.

    Args:
        vehicle_id: Specific vehicle to hold
        route_id: Route the vehicle operates on
        station: Station where the vehicle should hold
        hold_minutes: How many minutes to hold
        reason: Operational justification for the hold

    Returns:
        Hold instruction confirmation
    """
    action_record = {
        "action_id": f"HOLD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{vehicle_id}",
        "type": "vehicle_hold",
        "vehicle_id": vehicle_id,
        "route_id": route_id,
        "station": station,
        "hold_minutes": hold_minutes,
        "reason": reason,
        "status": "dispatched",
        "created_at": datetime.now().isoformat()
    }
    _action_log.append(action_record)

    return {
        "success": True,
        "action_id": action_record["action_id"],
        "message": f"Vehicle {vehicle_id} instructed to hold {hold_minutes} min at {station}",
        "reason": reason
    }


@tool
def get_action_log(limit: int = 30) -> list[dict]:
    """
    Retrieve recent transit management actions for audit and review.

    Args:
        limit: Maximum number of recent actions to return

    Returns:
        List of recent action records
    """
    return _action_log[-limit:]
