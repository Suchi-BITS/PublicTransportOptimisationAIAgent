# config/settings.py
# Public Transport Optimizer AI Agent v2 — Reactive Agent

import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class TransitConfig:
    openai_api_key: str  = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model_name:     str  = "gpt-4o"
    temperature:    float = 0.05

    network_name:   str  = "MetroLink Transit Authority"
    city:           str  = "Metro City"

    routes: List[str] = field(default_factory=lambda: [
        "R1-Red", "R2-Blue", "R3-Green", "R4-Yellow", "R5-Express",
    ])

    # Reactive thresholds — agent fires an optimisation pass when any is breached
    delay_threshold_minutes:     float = 5.0
    occupancy_threshold_percent: float = 85.0
    headway_deviation_pct:       float = 20.0
    incident_response_minutes:   float = 3.0

    planning_horizon_minutes: int = 60

    disclaimer: str = (
        "AI-generated transit optimisation recommendations. "
        "Safety-critical decisions require human dispatcher approval."
    )


transit_config = TransitConfig()
