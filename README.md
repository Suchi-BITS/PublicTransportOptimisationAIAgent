# AI Public Transportation Optimization Agent System

An agentic AI system built with LangGraph and LangChain that continuously monitors
passenger demand, traffic conditions, and scheduled events to dynamically adjust
bus and metro schedules in real time.

## Architecture Overview

Seven specialized agents collaborate via a shared LangGraph StateGraph:

```
[SUPERVISOR] (init)
      |
      v
[DEMAND AGENT] --> [TRAFFIC AGENT] --> [EVENT AGENT] --> [FLEET AGENT]
                                                               |
                                                               v
                                                   [SCHEDULE OPTIMIZER]
                                                               |
                                                               v
                                                        [ALERT AGENT]
                                                               |
                                                               v
                                                  [SUPERVISOR] (synthesis)
                                                               |
                                                               v
                                                             END
```

## Agent Descriptions

### Supervisor Agent
Entry point and final synthesizer. Routes the graph to the first monitoring agent,
then waits for all agents to complete before producing the final operations report
and computing the network status (CRITICAL / DISRUPTED / DEGRADED / NORMAL / ENHANCED).

### Demand Monitoring Agent
- Reads Automated Passenger Counter (APC) and fare system data
- Computes load factors for every bus and metro route
- Flags overcrowded routes (>90% capacity) and underutilized ones (<30%)
- Computes demand index: current ridership vs historical baseline
- Forecasts demand 1-3 hours ahead for proactive scheduling

### Traffic Monitoring Agent
- Reads traffic management API data for all bus corridors
- Calculates schedule deviation (extra minutes) per corridor
- Identifies active incidents: accidents, road works, event closures, weather
- Assesses compound risk: heavy traffic + overcrowding scenarios
- Metro lines are unaffected by road traffic but monitored for platform crowding

### Event Impact Agent
- Reads the events calendar for the planning horizon (default 6 hours)
- Estimates transit demand per event based on attendance and modal share
- Classifies events by impact: critical (5000+ transit pax), major, moderate
- Determines demand pattern: pre-event surge, post-event surge, or steady
- Recommends extra service types (shuttle, express, extended hours)

### Fleet Status Agent
- Polls AVL/GPS positions and on-time status for all active vehicles
- Detects bunching (vehicles too close) and gaps (headways too wide)
- Flags mechanical failures and out-of-service vehicles
- Monitors driver shift hours to avoid fatigue violations
- Calculates network-wide on-time performance percentage

### Schedule Optimizer Agent
- Synthesizes all four monitoring analyses in a single pass
- Uses LangChain `bind_tools()` with an agentic loop (up to 3 rounds) for:
  - `adjust_route_frequency`: Change headways, deploy or recall vehicles
  - `deploy_extra_service`: Add express trips, shuttles, event service
  - `reallocate_fleet`: Transfer vehicles from underused to overcrowded routes
- Prioritizes interventions: critical overcrowding first, proactive second, efficiency third

### Passenger Alert Agent
- Generates public-facing service alerts for all significant changes
- Calibrates severity: critical / major / minor / info
- Writes channel-appropriate copy: app, website, station displays, social media, SMS
- Issues positive event service notifications alongside disruption alerts
- Uses LangChain tool calling to publish via `issue_service_alert`

## Project Structure

```
transit_agents/
|-- main.py                         # Entry point, runs full optimization cycle
|-- requirements.txt
|-- .env.example
|
|-- config/
|   |-- settings.py                 # Network config, thresholds, fleet inventory
|
|-- data/
|   |-- models.py                   # Pydantic models: AgentState, PassengerDemandData, etc.
|
|-- agents/
|   |-- supervisor_agent.py         # Graph orchestrator and report synthesizer
|   |-- demand_agent.py             # Passenger load and ridership monitoring
|   |-- traffic_agent.py            # Road conditions and schedule adherence
|   |-- event_agent.py              # Events calendar and demand forecasting
|   |-- fleet_agent.py              # Vehicle tracking and mechanical status
|   |-- schedule_optimizer.py       # Schedule adjustments via LangChain tool calls
|   |-- alert_agent.py              # Passenger communications via tool calls
|
|-- tools/
|   |-- sensor_tools.py             # Data collection: demand, traffic, events, fleet
|   |-- action_tools.py             # Actions: frequency, extra service, alerts, realloc
|
|-- graph/
|   |-- transit_graph.py            # LangGraph StateGraph definition and compilation
```

## Key Data Models

```python
# Core state shared across all agents
class AgentState(BaseModel):
    demand_data: list[PassengerDemandData]
    traffic_data: list[TrafficConditionData]
    event_data: list[EventData]
    vehicle_statuses: list[VehicleStatus]
    
    demand_analysis: Optional[str]        # From DemandAgent
    traffic_analysis: Optional[str]       # From TrafficAgent
    event_analysis: Optional[str]         # From EventAgent
    fleet_analysis: Optional[str]         # From FleetAgent
    
    schedule_adjustments: list[ScheduleAdjustment]   # From Optimizer
    fleet_deployments: list[FleetDeploymentPlan]     # From Optimizer
    service_alerts: list[ServiceAlert]               # From AlertAgent
    
    optimization_plan: Optional[TransitOptimizationSchedule]  # From Supervisor
    current_agent: str                    # LangGraph routing control
```

## Setup and Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add OPENAI_API_KEY
```

### 3. Run optimization cycle
```bash
python main.py
```

## Network Configuration

Edit `config/settings.py` to match your transit network:

```python
transit_config = TransitConfig(
    network_name="City Transit Authority",
    city="Your City",
    bus_routes=["B1", "B2", "B3", "B10", "B22"],
    metro_lines=["Red", "Blue", "Green"],
    total_buses=120,
    total_metro_trains=30,
    bus_capacity=60,
    metro_capacity=400,
    overcrowding_threshold=0.90,        # 90% load factor = overcrowded
    underutilization_threshold=0.30,    # 30% load factor = underused
    planning_horizon_hours=6,
    on_time_target_percent=85.0
)
```

## Production Integration

Replace simulated tool implementations with real data sources:

| Tool | Production Data Source |
|------|----------------------|
| `fetch_passenger_demand` | AFC system API, APC sensor aggregator |
| `fetch_traffic_conditions` | Traffic management centre API, HERE/TomTom |
| `fetch_events_calendar` | City events permit API, venue ticketing systems |
| `fetch_fleet_status` | AVL/GPS system, CAD software API |
| `adjust_route_frequency` | CAD dispatch system API |
| `deploy_extra_service` | Crew management system + CAD |
| `issue_service_alert` | GTFS-RT alert feed, CMS for displays, Twilio for SMS |

## Continuous Operation

To run as a continuously monitoring system:

```python
import time
from main import run_optimization_cycle

while True:
    run_optimization_cycle()
    time.sleep(300)  # Re-optimize every 5 minutes
```

For production, use a proper scheduler (APScheduler, Celery beat, cron)
with database-backed state persistence and alerting on agent failures.
