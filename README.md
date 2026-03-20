# Public Transport Optimizer AI Agent v2

A reactive AI agent system built with LangGraph and LangChain that continuously
monitors a public transit network, detects threshold breaches across all routes and
vehicles, and generates targeted optimisation actions in real time. The system reacts
only when conditions exceed defined operating parameters — no unnecessary LLM calls
are made when the network is running normally.

The system runs fully end-to-end in demo mode with no external dependencies, and
upgrades automatically to live GPT-4o reasoning when an OpenAI API key is present.

---

## Table of Contents

1. [Use Case and Problem Statement](#use-case-and-problem-statement)
2. [System Architecture](#system-architecture)
3. [Architectural Pattern — Reactive Agent](#architectural-pattern)
4. [Trigger and Response Catalogue](#trigger-and-response-catalogue)
5. [Agent Graph Topology](#agent-graph-topology)
6. [Step-by-Step Workflow](#step-by-step-workflow)
7. [File-by-File Explanation](#file-by-file-explanation)
8. [Data Layer and Simulation Design](#data-layer-and-simulation-design)
9. [Network Configuration](#network-configuration)
10. [Production Deployment Guide](#production-deployment-guide)
11. [Run Modes and Quick Start](#run-modes-and-quick-start)
12. [Sample Output](#sample-output)
13. [Disclaimer](#disclaimer)

---

## Use Case and Problem Statement

### The Domain

Urban public transit networks operate dozens of routes with hundreds of vehicles
running on schedules designed for average conditions. Reality deviates from the
schedule constantly: road closures create cascading delays, stadium events create
sudden demand spikes on specific lines, breakdowns strand vehicles mid-route, and
traffic congestion degrades headways network-wide. Human dispatchers can monitor
one or two crisis situations simultaneously; they cannot track 16 vehicles across
5 routes continuously.

### The Problem

Three specific gaps make this an appropriate reactive agent problem.

First, most of the time nothing needs changing. In a typical 8-hour operating day,
90 percent of the time the network runs within acceptable parameters and no
intervention is needed. A system that runs a full planning cycle every 5 minutes
wastes compute and introduces unnecessary actions. A reactive agent acts only when
a threshold is breached, keeping the system efficient.

Second, when something does need changing, the response needs to be fast and targeted.
A 17-minute delay on one vehicle on one route requires a specific targeted action
(skip non-essential stops, hold connecting services downstream) not a full network
replan. The reactive pattern matches this requirement: observe, detect the specific
breach, choose from a catalogue of pre-defined responses.

Third, the number of possible states is large but the number of response types is
small. There are roughly 15 distinct response actions that cover all transit
operational scenarios (deploy extra vehicle, reroute around incident, skip stops,
activate bus priority signals, and so on). An agent that recognises which of these
actions fits the current trigger is more reliable than one that reasons from scratch
about what to do.

### What This System Delivers

For each invocation, the system produces: a network status label, a list of all
detected triggers sorted by severity, a set of route-specific optimisation actions
with rationale, and a fleet summary showing on-time rate and average occupancy. If
no thresholds are breached, the system confirms normal status and exits in under one
second.

---

## System Architecture

```
+------------------------------------------------------------------+
|  DATA LAYER                                                      |
|                                                                  |
|  get_fleet_positions()    GTFS-RT VehiclePositions (production)  |
|  get_incidents()          CAD (Computer-Aided Dispatch) API      |
|  get_passenger_demand()   APC (Automatic Passenger Counter) API  |
|  get_traffic_conditions() HERE Traffic API / TomTom Flow API     |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  REACTIVE AGENT CORE (agents/reactive_agent.py)                 |
|                                                                  |
|  1. OBSERVE                                                      |
|     Read all four data streams simultaneously                    |
|     16 vehicles, incidents, demand per route, traffic network    |
|                                                                  |
|  2. DETECT                                                       |
|     Scan all observations against 5 trigger types:              |
|       delay         (vehicle > 5min late)                        |
|       overcrowding  (vehicle > 85% capacity)                     |
|       incident      (from CAD feed)                              |
|       high_wait     (avg passenger wait > 12min)                 |
|       congestion    (network avg speed severely reduced)         |
|     Classify each trigger as severity: high or medium            |
|                                                                  |
|  3. REASON                                                       |
|     Group triggers by route                                      |
|     For each affected route: LLM selects best response from      |
|     catalogue of 15 pre-defined actions                          |
|     (or demo: auto-select first matching action)                 |
|                                                                  |
|  4. ACT                                                          |
|     Log recommended actions with route, severity, rationale      |
|                                                                  |
|  5. REPORT                                                       |
|     Build network status summary                                 |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  LANGGRAPH REACTIVE GRAPH (graph/transit_graph.py)              |
|                                                                  |
|  START -> reactive_agent_node                                    |
|             -> should_rerun? -> reactive_agent_node (loop)       |
|                              -> END                              |
|                                                                  |
|  Reruns once if high-severity triggers remain (max 2 passes)     |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  OUTPUT LAYER                                                    |
|  Network status (normal / degraded / disrupted)                  |
|  Trigger list sorted by severity                                 |
|  Route-level optimisation actions with rationale                 |
|  Fleet summary: on-time count, overcrowded count, avg occupancy  |
+------------------------------------------------------------------+
```

---

## Architectural Pattern

### Reactive Agent

The reactive agent pattern is the simplest and most appropriate architecture for
systems where most invocations require no action and only specific threshold breaches
require LLM reasoning.

In the reactive pattern, the agent does not plan ahead, does not maintain a goal
hierarchy, and does not model future states. It observes the current world state,
compares it against a fixed set of threshold conditions, and for each breach selects
the best response from a pre-defined catalogue. The LLM is called only once per
affected route — not once per vehicle or per trigger.

This is appropriate for transit optimisation because:

The response actions are well-defined and exhaustive. Every operational scenario in
public transit has a standard response procedure. The agent does not need to invent
new actions; it needs to correctly match the current scenario to the right standard
response.

Latency matters. A transit dispatcher needs a response recommendation within 2 to
3 seconds of a trigger being detected. A full planning cycle with multiple LLM calls
per trigger would take 20 to 30 seconds. The reactive pattern keeps each invocation
to 1 LLM call per affected route.

False positive cost is low but false negative cost is high. If the agent detects a
trigger that turns out to be transient, the cost of the suggested action (holding
one connecting service for 2 minutes) is minimal. If the agent misses a genuine
cascade delay, the passenger cost is significant. The threshold-based detection
errs on the side of sensitivity.

### Why Not Event-Driven Streaming

For an MVP presentation, event-driven real-time streaming (consuming a GTFS-RT
Protobuf stream via Kafka or a WebSocket) adds significant infrastructure complexity
(Kafka cluster, stream processing framework, GTFS-RT decoder) without changing the
core agent logic. The reactive agent produces the same reasoning quality when invoked
on a polling interval (e.g., every 30 seconds) as it would when triggered by a
streaming event. The data layer can be upgraded to streaming in production by
replacing `get_fleet_positions()` with a GTFS-RT consumer without touching any
agent or graph code.

---

## Trigger and Response Catalogue

The agent detects five trigger types and selects responses from a pre-defined catalogue.

### Trigger Types

| Trigger | Threshold | Severity | Detection Source |
|---|---|---|---|
| `delay` | vehicle > 5 minutes late | high if > 15min, medium otherwise | GTFS-RT positions |
| `overcrowding` | vehicle > 85% capacity | high if > 100%, medium otherwise | APC sensors |
| `incident` | any CAD-reported incident | inherits CAD severity | CAD system |
| `high_wait` | avg wait > 12 minutes at a stop | medium | APC + schedule |
| `network_congestion` | overall congestion high or severe | high or medium | traffic API |

### Response Catalogue

| Trigger | Available Responses |
|---|---|
| `delay` | skip_non_essential_stops, hold_connecting_services, deploy_relief_vehicle, update_passenger_information, alert_downstream_stops |
| `overcrowding` | deploy_extra_vehicle, increase_service_frequency, advise_passengers_next_service, short_turn_vehicle_for_extra_trip |
| `incident` | reroute_affected_vehicles, activate_diversion_route, deploy_bridge_service, suspend_route_segment, notify_all_affected_passengers |
| `high_wait` | increase_frequency, deploy_additional_vehicle, split_route_for_higher_frequency |
| `network_congestion` | activate_bus_priority_signals, advise_alternate_routes_to_passengers, adjust_headways_network_wide |

The LLM receives the trigger details, the current vehicle data for the affected route,
and the catalogue of available responses. It selects the single best action and
provides a two-sentence rationale. In demo mode the first catalogued action for the
highest-severity trigger type is selected deterministically.

---

## Agent Graph Topology

```
START
  |
  v
reactive_agent_node (pass 1)
  | Observe -> Detect -> Reason -> Act -> Report
  | sets state["network_status"], state["actions"], state["triggers"]
  |
  v
should_rerun()
  | if pass_count < 2 AND high-severity triggers remain: "rerun"
  | else: "end"
  |
  +-- "rerun" --> reactive_agent_node (pass 2: refine high-severity actions)
  |
  +-- "end"   --> END
```

The graph allows at most two passes. The second pass is triggered only when
high-severity triggers remain after the first pass, giving the agent an opportunity
to refine or escalate its response recommendations. For most invocations the system
completes in a single pass.

---

## Step-by-Step Workflow

### Step 1: Observe

The reactive agent calls all four data functions simultaneously (sequentially in
Python, but logically independent):

`get_fleet_positions()` returns 16 vehicles across 5 routes. Each vehicle record
includes: vehicle ID, route, current stop, next stop, delay in minutes, passenger
load, capacity, occupancy percent, speed, vehicle type, and operational status.

`get_incidents()` returns 0 to 3 active incidents from the simulated CAD feed. Each
incident record includes: incident type, affected route and stop, severity, and
estimated duration.

`get_passenger_demand()` returns per-route demand metrics: current boardings per
hour, peak stop, average wait time, and demand trend (increasing, stable, decreasing).

`get_traffic_conditions()` returns the overall network congestion level, average
network speed, and a list of active traffic hotspots with their locations, delay
contribution, and cause.

### Step 2: Detect

The `_detect_triggers()` function iterates through all vehicle records, incidents,
demand records, and the traffic summary, comparing each value against the thresholds
configured in `TransitConfig`. It returns a flat list of trigger dicts, each containing
type, severity, affected vehicle or route, the threshold-breaching value, and a
human-readable detail string.

Triggers are classified as high or medium severity. High severity triggers are those
where the breach is significantly above threshold (delay > 15 minutes, occupancy > 100
percent, traffic severity "severe") or the incident severity is high.

### Step 3: Reason

Triggers are grouped by route. For each affected route the `_reason_for_route()`
function builds a prompt containing: the trigger details, the current vehicle data
for that route, and the list of available response actions for the trigger types
present. The LLM (or demo auto-select) chooses the single best action and produces
a rationale.

Grouping by route ensures the LLM reasons about the full context on a given route
(multiple delayed and overcrowded vehicles, combined with an incident) rather than
treating each trigger in isolation.

### Step 4: Act

The action records are accumulated in state with their route, trigger types, severity,
recommended action, rationale, and timestamp. These are what the human dispatcher
sees in the final output.

### Step 5: Report

The network status is computed deterministically from the trigger severity counts:
`disrupted` if two or more high-severity triggers exist, `degraded` if one or more
high-severity triggers, `normal` if no high-severity triggers and the system found
nothing to flag. The report is built with fleet summary statistics and the trigger
and action tables.

---

## File-by-File Explanation

### main.py

Entry point. Creates the initial state dict. Attempts to build and invoke the LangGraph
graph. Falls back to direct `run_reactive_agent()` call if LangGraph is not installed.
`print_report()` formats and prints: network status, triggers sorted by severity,
optimisation actions with rationale, and fleet summary statistics.

### config/settings.py

`TransitConfig` dataclass. Holds: OpenAI API key, network name and city, list of 5
route identifiers, reactive thresholds (delay at 5 minutes, occupancy at 85 percent,
wait at 12 minutes configured implicitly in agent code), and planning horizon.

### data/simulation.py

Four simulation functions. `get_fleet_positions()` generates 2 to 4 vehicles per
route with day-seeded random delays and occupancy values. `get_incidents()` generates
0 to 3 incidents per invocation. `get_passenger_demand()` generates route-level
demand metrics. `get_traffic_conditions()` generates a network congestion state and
hotspot list. All use the date-seeded `_rng(salt)` pattern.

Vehicle capacity is specified per vehicle type: articulated bus 120, standard bus 75,
minibus 40, express coach 55. Occupancy percent is computed from load divided by
capacity and can exceed 100 percent to simulate overcrowding events.

### agents/base.py

Standard `_demo_mode()` and `call_llm()` functions shared by the reactive agent.

### agents/reactive_agent.py

The core reactive agent. `run_reactive_agent(state)` orchestrates the full
Observe-Detect-Reason-Act-Report cycle. `_detect_triggers()` applies threshold logic
to all observations. `_reason_for_route()` calls the LLM for each affected route.
`RESPONSE_OPTIONS` dict maps trigger types to their catalogue of available actions.
`_build_optimisation_report()` and `_build_normal_report()` produce the final output
text for disrupted and normal cases respectively.

### graph/transit_graph.py

`TransitState` TypedDict for all state keys. `reactive_node()` wraps
`run_reactive_agent()` with a pass counter. `should_rerun()` implements the
conditional self-loop: rerun if high-severity triggers remain and under the pass
limit (maximum 2). `build_transit_graph()` wires the single node with a conditional
self-edge.

---

## Data Layer and Simulation Design

The 5 routes use distinct vehicle types with different capacities and typical routes
through different parts of the city, giving each route a distinct character. R1-Red
uses articulated buses (120 capacity) on a high-demand central-to-airport corridor.
R4-Yellow uses minibuses (40 capacity) on a lower-demand local route. This means
overcrowding triggers occur more frequently on R4-Yellow even at lower absolute
passenger loads, which is realistic.

The incident generator uses a different RNG salt (500) from the vehicle position
generator (10 per route), ensuring incidents are independent of vehicle positions —
a route can have a delay and no incident, or an incident with on-time vehicles,
or both.

---

## Network Configuration

Five routes covering different corridor types.

| Route | Vehicle Type | Capacity | Stops |
|---|---|---|---|
| R1-Red | Articulated bus | 120 | Central, Eastgate, Stadium, Airport, Northpark |
| R2-Blue | Standard bus | 75 | Central, Westside, Hospital, University, Suburbs |
| R3-Green | Standard bus | 75 | Downtown, Market, Riverside, Industrial, Eastgate |
| R4-Yellow | Minibus | 40 | Airport, BusinessPark, Central, Southend, Port |
| R5-Express | Express coach | 55 | Central, Airport, University, Northpark |

Thresholds: delay alert at 5 minutes, overcrowding alert at 85 percent occupancy.
Re-run trigger: 2 or more high-severity triggers after pass 1.

---

## Production Deployment Guide

### Replacing Fleet Position Data

Replace `get_fleet_positions()` with a GTFS-RT VehiclePositions feed consumer.
GTFS-RT is a Protocol Buffer format — use the `gtfs-realtime-bindings` Python library
to decode the feed, then map each `VehiclePosition` entity to the vehicle dict schema
expected by `_detect_triggers()`. Most transit agencies publish their GTFS-RT feeds
publicly; the feed URL is usually in the agency's developer portal.

### Replacing Incident Data

Replace `get_incidents()` with a read from the agency's CAD (Computer-Aided Dispatch)
system API. For smaller agencies, incidents can be sourced from the GTFS-RT
`Alerts` feed, which carries service alerts in a structured format. Map each alert's
affected entity (route, stop, trip) to the incident dict schema.

### Replacing Passenger Demand Data

Replace `get_passenger_demand()` with reads from APC (Automatic Passenger Counter)
sensors. Most modern transit vehicles carry APC systems that transmit boardings and
alightings per stop via cellular to a central server. API access depends on the vendor
(typically Clever Devices or Luminator). Alternatively, use GTFS-RT `TripUpdates`
feeds which carry predicted arrival times and can be used to compute headway deviations.

### Replacing Traffic Data

Replace `get_traffic_conditions()` with HERE Routing API or TomTom Traffic Flow API.
Both provide real-time speed data by road segment, which can be used to compute
congestion level and estimate delay contributions on specific transit corridors.

### Upgrading to Streaming

To upgrade from polling to streaming, wrap the `run_reactive_agent()` call in a Kafka
consumer loop that triggers the agent on each GTFS-RT feed update. The agent code
itself does not change. The polling interval (30 seconds for MVP) is simply replaced
by an event trigger from the Kafka topic.

---

## Run Modes and Quick Start

### Demo Mode

```bash
cd transit_v2
python main.py
```

### Live LLM Mode

```bash
pip install langgraph langchain-core langchain-openai python-dotenv
cp .env.example .env
# Add OPENAI_API_KEY=sk-your-key to .env
python main.py
```

---

## Sample Output

```
PUBLIC TRANSPORT OPTIMIZER AI AGENT v2
Architecture: Reactive Agent
Network: MetroLink Transit Authority
Mode: DEMO (no API key)

[REACTIVE AGENT] Starting transit network observation...
  Observed: 16 vehicles, 2 incident(s), congestion=low
  Detected: 12 trigger(s) — 3 high, 9 medium
  Actions recommended: 5 | Status: DISRUPTED

NETWORK STATUS: DISRUPTED

TRIGGERS DETECTED (12):
  [HIGH  ] Vehicle R2-Blue-V04 is 17.8min late at Central
  [HIGH  ] Traffic Congestion on R3-Green at Eastgate — est. 7min
  [HIGH  ] Road Closure on R1-Red at Northpark — est. 33min
  [MEDIUM] Vehicle R1-Red-V01 at 100.0% capacity on R1-Red
  [MEDIUM] Vehicle R2-Blue-V02 is 5.3min late at Hospital
  [MEDIUM] Average wait 14.5min on R5-Express at University

OPTIMISATION ACTIONS (5):

  Route: R2-Blue | Severity: HIGH
  Action: SKIP NON ESSENTIAL STOPS
  Rationale: Vehicle V04 is 17.8 minutes late. Skipping low-boarding
  intermediate stops will recover approximately 4 minutes.

  Route: R1-Red | Severity: HIGH
  Action: ACTIVATE DIVERSION ROUTE
  Rationale: Road closure at Northpark — divert via Industrial Bridge.

FLEET SUMMARY:
  16 vehicles | On-time: 9 | Overcrowded: 2 | Avg occupancy: 71.8%
```

---

## Disclaimer

AI-generated transit optimisation recommendations. Safety-critical decisions,
including route suspensions and emergency diversions, require human dispatcher
approval before implementation. This system provides decision support only and
does not have authority to issue operational commands to vehicles or infrastructure.
