# data/models.py
# Shared data models for the transit optimization system

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, time


class PassengerDemandData(BaseModel):
    """Real-time and forecast passenger demand by route and station."""
    timestamp: datetime = Field(default_factory=datetime.now)
    route_id: str
    route_type: Literal["bus", "metro"]
    current_passengers: int
    vehicle_capacity: int
    load_factor: float = Field(description="current_passengers / vehicle_capacity, 0.0-1.0+")
    demand_forecast_1h: int = Field(description="Expected passengers in next 1 hour")
    demand_forecast_3h: int = Field(description="Expected passengers in next 3 hours")
    peak_station: str = Field(description="Station with highest demand on this route")
    peak_station_wait_minutes: float = Field(description="Current average wait at peak station")
    baseline_demand: int = Field(description="Historical average for this time/day")
    demand_index: float = Field(description="current vs baseline ratio, 1.0 = normal")


class TrafficConditionData(BaseModel):
    """Traffic and road condition data affecting bus operations."""
    timestamp: datetime = Field(default_factory=datetime.now)
    corridor: str = Field(description="Road corridor or transit segment identifier")
    route_ids_affected: list[str]
    current_speed_kmh: float
    free_flow_speed_kmh: float
    congestion_level: Literal["free_flow", "light", "moderate", "heavy", "gridlock"]
    delay_minutes: float = Field(description="Extra minutes added to scheduled journey time")
    incident_type: Optional[Literal["accident", "road_works", "event_closure", "weather", "none"]] = "none"
    incident_description: Optional[str] = None
    estimated_clearance_time: Optional[str] = None
    affected_stations: list[str] = Field(default_factory=list)


class EventData(BaseModel):
    """Scheduled events that will impact transit demand."""
    event_id: str
    event_name: str
    venue: str
    nearest_station: str
    event_type: Literal[
        "sports", "concert", "conference", "festival",
        "public_gathering", "market", "parade", "other"
    ]
    start_time: datetime
    end_time: datetime
    expected_attendance: int
    transit_modal_share: float = Field(
        description="Fraction of attendees expected to use transit, 0.0-1.0"
    )
    estimated_transit_demand: int = Field(
        description="expected_attendance * transit_modal_share"
    )
    affected_routes: list[str] = Field(default_factory=list)
    demand_pattern: Literal["surge_before", "surge_after", "both", "steady"] = "both"
    requires_special_service: bool = False


class VehicleStatus(BaseModel):
    """Real-time status of individual vehicles in the fleet."""
    vehicle_id: str
    vehicle_type: Literal["bus", "metro"]
    route_id: str
    current_location: str
    next_stop: str
    passengers_onboard: int
    capacity: int
    on_time_status: Literal["on_time", "early", "late", "very_late"]
    delay_minutes: float = 0.0
    fuel_level_percent: Optional[float] = None
    mechanical_status: Literal["operational", "degraded", "out_of_service"] = "operational"
    driver_hours_on_shift: float = 0.0


class ScheduleAdjustment(BaseModel):
    """A schedule modification action for a route."""
    route_id: str
    route_type: Literal["bus", "metro"]
    adjustment_type: Literal[
        "increase_frequency", "decrease_frequency",
        "deploy_extra_vehicle", "recall_vehicle",
        "extend_route", "short_turn", "express_service",
        "skip_stop", "hold_at_station", "cancel_trip"
    ]
    affected_direction: Literal["inbound", "outbound", "both"] = "both"
    new_headway_minutes: Optional[int] = None
    vehicles_added: int = 0
    vehicles_removed: int = 0
    effective_from: Optional[datetime] = None
    effective_until: Optional[datetime] = None
    affected_stops: list[str] = Field(default_factory=list)
    priority: Literal["critical", "high", "normal", "low"] = "normal"
    estimated_capacity_gain: int = 0
    reason: str
    passenger_communication_required: bool = False


class FleetDeploymentPlan(BaseModel):
    """Fleet reallocation and deployment strategy."""
    route_id: str
    route_type: Literal["bus", "metro"]
    current_vehicles: int
    recommended_vehicles: int
    vehicles_to_add: int = 0
    vehicles_to_withdraw: int = 0
    source_routes: list[str] = Field(
        default_factory=list,
        description="Routes from which vehicles are being redeployed"
    )
    deployment_time: Optional[datetime] = None
    justification: str
    expected_load_factor_after: float


class ServiceAlert(BaseModel):
    """Public-facing service alert for passengers."""
    alert_id: str
    severity: Literal["critical", "major", "minor", "info"]
    alert_type: Literal[
        "delay", "disruption", "route_change", "extra_service",
        "cancellation", "event_service", "general"
    ]
    affected_routes: list[str]
    affected_stations: list[str]
    headline: str
    body: str
    alternative_routes: list[str] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    channels: list[str] = Field(
        default_factory=lambda: ["app", "website", "station_displays", "social_media"]
    )


class TransitOptimizationSchedule(BaseModel):
    """Final consolidated transit optimization output."""
    generated_at: datetime = Field(default_factory=datetime.now)
    planning_horizon_hours: int
    schedule_adjustments: list[ScheduleAdjustment] = Field(default_factory=list)
    fleet_deployments: list[FleetDeploymentPlan] = Field(default_factory=list)
    service_alerts: list[ServiceAlert] = Field(default_factory=list)
    network_status: Literal["critical", "disrupted", "degraded", "normal", "enhanced"] = "normal"
    estimated_passengers_served: int = 0
    estimated_capacity_utilized_percent: float = 0.0
    summary: str = ""
    kpi_projections: dict = Field(default_factory=dict)


class AgentState(BaseModel):
    """
    Shared state flowing through the LangGraph agent graph.
    Each agent reads from and writes to this state.
    """
    # Raw sensor/feed data collected by monitoring agents
    demand_data: list[PassengerDemandData] = Field(default_factory=list)
    traffic_data: list[TrafficConditionData] = Field(default_factory=list)
    event_data: list[EventData] = Field(default_factory=list)
    vehicle_statuses: list[VehicleStatus] = Field(default_factory=list)

    # Analysis outputs from monitoring agents
    demand_analysis: Optional[str] = None
    traffic_analysis: Optional[str] = None
    event_analysis: Optional[str] = None
    fleet_analysis: Optional[str] = None

    # Decisions from planning agents
    schedule_adjustments: list[ScheduleAdjustment] = Field(default_factory=list)
    fleet_deployments: list[FleetDeploymentPlan] = Field(default_factory=list)
    service_alerts: list[ServiceAlert] = Field(default_factory=list)

    # Final consolidated optimization plan
    optimization_plan: Optional[TransitOptimizationSchedule] = None

    # Control flow
    current_agent: str = "supervisor"
    iteration_count: int = 0
    errors: list[str] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
