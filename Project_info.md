# Asphr — Project Information

---

## 1. Problem We Are Solving

Urban road networks in cities like Mumbai are plagued by a combination of issues that existing navigation tools fail to address holistically:

- **Road hazards are invisible to current routers.** Potholes, speed bumps, craters, and deteriorated surfaces cause vehicle damage, rider discomfort, and accidents — yet Google Maps and similar tools route without any awareness of pavement quality.
- **Traffic data is reactive, not predictive.** Current solutions rely on historical averages or delayed crowdsourced data, missing real-time congestion shifts that develop within minutes.
- **One-size-fits-all routing ignores vehicle type.** A bike, a truck, and a supercar have fundamentally different road constraints. No mainstream tool accounts for this at the routing algorithm level.
- **No closed-loop learning from drivers.** Drivers have no way to contribute structured feedback that directly improves future routing decisions for others.
- **IoT telemetry from vehicles is unused.** Accelerometer and gyroscope data from vehicle fleets contains rich pavement condition signals that are currently discarded rather than fused into road intelligence.

---

## 2. Our Proposed Solution

Asphr is a **full-stack intelligent dynamic routing engine** that fuses real-time IoT sensor data, machine learning inference, and spatial graph algorithms to compute personalized, hazard-aware routes across Mumbai's road network.

The system:
1. **Ingests IoT telemetry** (GPS + accelerometer/gyroscope) from a vehicle fleet, classifying pavement condition in real time and snapping readings to road segments via PostGIS spatial queries.
2. **Runs two ML models concurrently** — a PyTorch LSTM that forecasts road speeds 30 minutes ahead, and a Scikit-Learn ensemble that predicts per-segment hazard probability scores.
3. **Enriches a live in-memory graph** of Mumbai's road network (85k+ OSM nodes) with composite edge weights derived from traffic, hazard scores, weather, and IoT aggregates — refreshed every 5 minutes.
4. **Computes multi-objective routes** tuned to the driver's goal (fastest, safest, straightest, or popular) and vehicle profile (bike, car, truck, supercar), using Dijkstra and custom A* algorithms on the enriched graph.
5. **Broadcasts live hazard alerts and weather updates** to connected clients via WebSockets, enabling real-time awareness without page refreshes.
6. **Collects RLHF-style feedback** after each journey (5 structured questions covering accuracy, comfort, unmapped hazards, efficiency, and recommendation) to create a self-improving training loop.

---

## 3. Key Features / USP

### Multi-Objective Routing
Four distinct routing strategies with mathematically distinct edge weight formulations:
- **Fastest** — travel time minimization with a proactive congestion penalty derived from the LSTM traffic forecaster.
- **Safest** — hazard-weighted path that blends ML hazard prediction (70%) with live DB scores (30%), scaled further by weather severity.
- **Straightest** — custom A* with angular bearing deviation cost, minimizing turns for the smoothest possible path.
- **Popular** — scenic routing that rewards segments near high-density Points of Interest.

### Vehicle-Aware Graph Filtering
Before pathfinding, the graph is dynamically pruned per vehicle type:
- **Bike** — strips out motorways and trunk roads.
- **Truck** — removes paths narrower than 3m and residential/service streets.
- **Supercar** — excludes any segment with a speed bump flag or unclassified surface type.
- **Car** — unrestricted routing.

### Real-Time IoT Hazard Pipeline
Accelerometer readings are classified into pavement condition tiers (smooth / moderate / rough / severe), snapped to road segments, and immediately broadcast as `hazard_alert` WebSocket events to all connected clients. Hazard records carry a TTL and auto-expire after 2 hours.

### Machine Learning Inference Layer
- **Traffic Forecaster**: PyTorch LSTM (4-step temporal window, 45-minute lookback) predicts segment speeds 30 minutes ahead. Integrated directly into the Fastest route weight formula.
- **Hazard Predictor**: Scikit-Learn Gradient Boosting model on a 23-feature vector (vibration magnitude, traffic speed, congestion level, road type, weather condition, temporal signals). Supports single-segment and batch inference.

### Live Hazard Heatmap (Maps Page)
Viewport-bound spatial query returns all road segments with hazard scores, rendered as a color-ramp GeoJSON layer (green → yellow → orange → red) on an interactive Mapbox map.

### RLHF Feedback Loop
Post-route feedback collected across 5 structured dimensions: hazard accuracy, ride comfort, unmapped hazard encounters, route efficiency, and overall recommendation score. Stored as spatial records (start/end/route geometry) for future model fine-tuning.

### SOS Alert System
Gyroscope-detected accident events trigger SOS alerts stored in the database with hospital notification flags, enabling emergency response integration.

### Background Scheduler
APScheduler runs three recurring jobs:
- Graph weight enrichment every 5 minutes.
- Weather grid refresh from OpenWeatherMap every 10 minutes.
- TTL-expired hazard cleanup every 5 minutes.

---

## 4. Tech Stack

### Frontend
| Layer | Technology |
|---|---|
| Framework | React 19 + Vite (SPA) |
| Map Rendering | Mapbox GL JS 3.25 + react-map-gl |
| Routing | React Router v7 |
| Styling | Tailwind CSS 3.4 + PostCSS |
| Icons | Lucide React |
| Linting | oxlint |

### Backend — API & Orchestration
| Layer | Technology |
|---|---|
| Framework | FastAPI (Python, async ASGI) |
| Server | Uvicorn |
| ORM | SQLAlchemy 2.0 (async) + GeoAlchemy2 |
| Database | Supabase PostgreSQL + PostGIS |
| Scheduling | APScheduler |
| HTTP Client | httpx (async) |
| Config | pydantic-settings + .env |
| Containerization | Docker |

### Backend — Routing & Spatial
| Layer | Technology |
|---|---|
| Graph Engine | NetworkX (in-memory MultiDiGraph) |
| OSM Data | OSMnx (download, simplify, cache as GraphML) |
| Spatial Math | Shapely (geometry operations) |
| Pathfinding | Dijkstra (NetworkX), custom A* (bearing-weighted) |
| Spatial DB Queries | PostGIS ST_Intersects, ST_DWithin, ST_AsText, GIST indexes |

### Backend — Machine Learning
| Layer | Technology |
|---|---|
| Traffic Forecaster | PyTorch (LSTM, 2-layer, hidden_dim=32) |
| Hazard Predictor | Scikit-Learn (Gradient Boosting, 23 features) |
| Model Serialization | joblib (.pkl) + PyTorch checkpoint (.pt) |
| Numerical Computing | NumPy, Pandas |

### External Integrations
| Service | Purpose |
|---|---|
| Mapbox Geocoding API | Forward/reverse geocoding (primary) |
| Nominatim (OSM) | Geocoding fallback with rate limiting |
| OpenWeatherMap API | Weather grid data (temperature, precipitation, visibility) |
| TomTom Traffic Flow API | Real-time congestion levels and segment speeds |

---

## 5. Architecture

### System Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│                                                                 │
│   React SPA (Mapbox GL)          IoT Telemetry Fleet            │
│   - Route search & map UI        - GPS + Accelerometer/Gyro     │
│   - Hazard heatmap overlay       - Vibration & road condition   │
│   - WebSocket listener           - SOS trigger                  │
└──────────────┬──────────────────────────┬───────────────────────┘
               │ HTTP REST                │ HTTP Ingest
               ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI API GATEWAY                          │
│                                                                 │
│  REST Routers          WebSocket Server     Background Tasks    │
│  /api/v1/routes        /ws                  Async DB writes     │
│  /api/v1/geocode       Broadcast pool       Feedback logging    │
│  /api/v1/iot           Hazard alerts                            │
│  /api/v1/custom-db     Weather updates                          │
│  /health                                                        │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CORE PROCESSING ENGINES                        │
│                                                                 │
│  GraphManager              RouteOptimizer                       │
│  - OSMnx download          - KD-Tree coordinate snapping        │
│  - GraphML caching         - Vehicle profile graph filtering    │
│  - DB sync & enrichment    - Dijkstra / custom A* pathfinding   │
│                            - Weight formula selection           │
│                                                                 │
│  HazardPredictor           TrafficForecaster                    │
│  - sklearn GBM             - PyTorch LSTM                       │
│  - 23-feature vectors      - 4-step temporal sequences          │
│  - Score: [0.0, 1.0]       - Predicts speed 30min ahead         │
│                                                                 │
│  GeocodingService          WeatherService                       │
│  - Mapbox → Nominatim      - OpenWeatherMap polling             │
│  - In-memory + DB cache    - 0.05° grid propagation             │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│              DATA & STORAGE LAYER                               │
│                                                                 │
│  Supabase PostgreSQL + PostGIS                                  │
│                                                                 │
│  road_segments       — OSM edges, LineString geometry           │
│  segment_hazards     — ML + live hazard scores, TTL             │
│  iot_readings        — Accel/gyro telemetry per segment         │
│  traffic_conditions  — Speeds, congestion levels (0–4)          │
│  weather_grid        — Spatial polygon cells, 0.05° resolution  │
│  popular_places      — POI coordinates for scenic routing       │
│  route_feedback      — RLHF training data (geometry + ratings)  │
│  sos_alerts          — Accident triggers, hospital notification  │
│  vehicle_profiles    — Per-type road constraint configurations   │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│             BACKGROUND SCHEDULER (APScheduler)                  │
│                                                                 │
│  Every 5 min:  Graph weight enrichment (ML inference + DB)      │
│  Every 10 min: Weather grid refresh (OpenWeatherMap)            │
│  Every 5 min:  TTL hazard expiration cleanup                    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Data Flows

**Route Computation Request**
```
Client → POST /api/v1/routes/compute
       → RouteService → RouteOptimizer
       → Snap coords to graph nodes (KD-Tree)
       → Filter graph by vehicle profile
       → Select edge weight attribute by route_type
       → nx.shortest_path (Dijkstra / A*)
       → Build GeoJSON response with instructions, hazard scores, weather alerts
       → Return to client
```

**IoT Hazard Ingestion**
```
IoT Device → POST /api/v1/iot
           → Parse accel/gyro, compute vibration magnitude
           → ST_DWithin snap to road_segments
           → Classify condition → write iot_readings
           → Trigger hazard_alert WebSocket broadcast to all clients
```

**Graph Enrichment (every 5 min)**
```
APScheduler → fetch segment hazards, traffic speeds, weather cells, IoT aggregates
            → build 23-feature vectors per edge
            → HazardPredictor.predict_batch() [sklearn]
            → TrafficForecaster.predict_batch() [PyTorch]
            → Apply weight formulas to all 4 routing objectives
            → Update in-memory graph edge attributes
            → Broadcast graph_refreshed via WebSocket
```

### Edge Weight Formulas

**Fastest:**
$$W = \frac{L}{S_{current}/3.6} \times \left(1 + \max\left(0, \frac{S_{limit} - S_{predicted}}{S_{limit}}\right)\right)$$

**Safest:**
$$W = L \times (1 + H_{blended}) \times (1 + C_{weather})$$
where $H_{blended} = 0.7 \cdot H_{ML} + 0.3 \cdot H_{DB}$ and $C_{weather} \in \{0.0, 0.2, 0.4\}$

**Straightest (A\*):**
$$\text{cost}(u,v) = L \times T_{traffic} \times (1 + \Delta\theta \times 2.5)$$

**Popular:**
$$W = \frac{L}{1 + \sum_{i \in POI_{500m}} \text{score}_i}$$

---

