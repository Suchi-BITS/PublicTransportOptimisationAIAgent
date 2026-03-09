# config/settings.py
# Transit Optimization System Configuration

from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


class TransitConfig(BaseModel):
    """Core configuration for the transit optimization system."""

    # LLM settings
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model_name: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2048

    # Transit network identity
    network_name: str = "MetroLink Transit Authority"
    city: str = "Metro City"
    timezone: str = "America/New_York"

    # Network topology
    bus_routes: list[str] = ["B1", "B2", "B3", "B4", "B5", "B6"]
    metro_lines: list[str] = ["M1", "M2", "M3"]
    key_stations: list[str] = [
        "Central Station", "Airport Terminal", "Stadium District",
        "University Hub", "Downtown Core", "Riverside", "Tech Park", "Westgate Mall"
    ]

    # Fleet inventory
    total_buses: int = 80
    total_metro_trains: int = 24
    bus_capacity: int = 60         # passengers per bus
    metro_capacity: int = 300      # passengers per train

    # Demand thresholds (load factor = passengers / capacity)
    overcrowding_threshold: float = 0.90    # 90% full triggers intervention
    underutilization_threshold: float = 0.30  # 30% full triggers reduction
    surge_multiplier_threshold: float = 1.5   # 50% above baseline is a surge

    # Scheduling parameters
    min_headway_minutes: int = 3      # minimum gap between vehicles
    max_headway_minutes: int = 30     # maximum acceptable gap
    planning_horizon_hours: int = 6   # how far ahead to plan

    # Performance targets
    on_time_target_percent: float = 85.0
    passenger_satisfaction_target: float = 4.0  # out of 5.0


transit_config = TransitConfig()
