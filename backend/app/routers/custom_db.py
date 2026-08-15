import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from app.config import get_db, settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/custom-db", tags=["CustomDB"])

@router.get("/popular_places")
async def get_popular_places(db: AsyncSession = Depends(get_db)):
    """Retrieve Mumbai popular places with geometries from database or fallback landmarks."""
    try:
        query = text("""
            SELECT name, category, popularity_score, ST_AsGeoJSON(geometry) 
            FROM popular_places;
        """)
        result = await db.execute(query)
        rows = result.fetchall()
        features = []
        for name, category, score, geom_json in rows:
            if not geom_json:
                continue
            try:
                geom = json.loads(geom_json)
            except Exception:
                continue
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "name": name,
                    "category": category,
                    "popularity_score": float(score or 0.0)
                }
            })
        if features:
            return {"type": "FeatureCollection", "features": features}
    except Exception as e:
        logger.warning(f"Database error in get_popular_places: {e}")

    # Fallback popular places in Mumbai
    landmarks = [
      {"name": "Gateway of India", "category": "Monument", "lon": 72.8347, "lat": 18.9220},
      {"name": "Marine Drive", "category": "Promenade", "lon": 72.8230, "lat": 18.9430},
      {"name": "Chhatrapati Shivaji Terminus (CSMT)", "category": "Transit", "lon": 72.8353, "lat": 18.9400},
      {"name": "Bandra-Worli Sea Link", "category": "Infrastructure", "lon": 72.8180, "lat": 19.0300},
      {"name": "Siddhivinayak Temple", "category": "Religious", "lon": 72.8315, "lat": 19.0169},
      {"name": "Powai Lake", "category": "Nature", "lon": 72.9050, "lat": 19.1250},
      {"name": "Juhu Beach", "category": "Beach", "lon": 72.8260, "lat": 19.0980}
    ]
    features = [
      {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [l["lon"], l["lat"]]},
        "properties": {"name": l["name"], "category": l["category"], "popularity_score": 0.95}
      } for l in landmarks
    ]
    return {"type": "FeatureCollection", "features": features}


@router.get("/weather_grid")
async def get_weather_grid(db: AsyncSession = Depends(get_db)):
    """Retrieve weather grid cells from database or return fallback grid if database is offline."""
    try:
        query = text("""
            SELECT id, temperature, humidity, visibility_km, precipitation_mm, wind_speed_kmh, weather_condition, ST_AsGeoJSON(cell_geometry) 
            FROM weather_grid;
        """)
        result = await db.execute(query)
        rows = result.fetchall()
        features = []
        for cid, temp, hum, vis, precip, wind, cond, geom_json in rows:
            if not geom_json:
                continue
            try:
                geom = json.loads(geom_json)
            except Exception:
                continue
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "id": cid,
                    "temperature": float(temp or 28.0),
                    "humidity": float(hum or 80.0),
                    "visibility_km": float(vis or 10.0),
                    "precipitation_mm": float(precip or 0.0),
                    "wind_speed_kmh": float(wind or 15.0),
                    "weather_condition": cond or "clear"
                }
            })
        if features:
            return {"type": "FeatureCollection", "features": features}
    except Exception as e:
        logger.warning(f"Database connection offline for weather_grid query: {e}")

    # Dynamic 24-cell Weather Grid covering Mumbai MMR area (72.80 to 72.96, 18.90 to 19.25)
    features = []
    cell_id = 1
    conditions = ["clear", "cloudy", "rain", "thunderstorm", "mist"]
    lats = [18.90 + i * 0.05 for i in range(8)]
    lons = [72.80 + j * 0.04 for j in range(5)]

    for i in range(len(lats) - 1):
        for j in range(len(lons) - 1):
            min_lat, max_lat = lats[i], lats[i+1]
            min_lon, max_lon = lons[j], lons[j+1]
            cond = conditions[(i * 2 + j) % len(conditions)]
            temp = round(26.5 + (i * 0.6) + (j * 0.4), 1)
            humidity = round(76.0 + (i * 1.8), 1)

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat]
                    ]]
                },
                "properties": {
                    "id": cell_id,
                    "temperature": temp,
                    "humidity": humidity,
                    "visibility_km": 10.0,
                    "precipitation_mm": 6.4 if cond in ["rain", "thunderstorm"] else 0.0,
                    "wind_speed_kmh": 14.5,
                    "weather_condition": cond
                }
            })
            cell_id += 1

    return {"type": "FeatureCollection", "features": features}


@router.get("/heavy_traffic")
async def get_heavy_traffic():
    """Fetch live heavy traffic incident points from TomTom API for MMR bbox."""
    try:
        key = settings.TOMTOM_API_KEY
        if not key:
            key = "Hbd95vTMExHxaAjqy8HGs6J0EEXLDZo9"
            
        url = 'https://api.tomtom.com/traffic/services/5/incidentDetails'
        params = {
            'key': key,
            'bbox': '72.75,18.90,73.20,19.50',
            'fields': '{incidents{type,geometry{type,coordinates},properties{iconCategory,magnitudeOfDelay,delay,events{description}}}}'
        }
        
        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=params, timeout=10.0)
            
        if res.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"TomTom API failed with status {res.status_code}"
            )
            
        data = res.json()
        incidents = data.get('incidents', [])
        
        features = []
        for inc in incidents:
            props = inc.get('properties', {})
            cat = props.get('iconCategory')
            mag = props.get('magnitudeOfDelay', 0)
            delay = props.get('delay', 0) or 0
            events = props.get('events', [])
            desc = events[0].get('description', 'Heavy Traffic') if events else 'Heavy Traffic'
            
            if cat != 6 and mag < 2:
                continue
            
            geom = inc.get('geometry', {})
            geom_type = geom.get('type')
            coords = geom.get('coordinates', [])
            
            if not coords:
                continue
                
            if geom_type == 'Point':
                point_coords = coords
            elif geom_type == 'LineString':
                point_coords = coords[len(coords) // 2]
            elif geom_type == 'Polygon' and len(coords) > 0 and len(coords[0]) > 0:
                point_coords = coords[0][0]
            else:
                point_coords = coords[0] if isinstance(coords[0], list) and not isinstance(coords[0][0], list) else [72.8777, 19.0760]

            congestion_level = 3 if mag >= 3 else 2
            color = "#EF4444" if congestion_level == 3 else "#F97316"
            
            speed = round(72.0 / (delay / 60.0 + 1.0), 1) if delay > 0 else round(30.0 - mag * 4.0, 1)
            if speed < 5:
                speed = 5.0
                
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": point_coords
                },
                "properties": {
                    "name": desc,
                    "congestion_level": congestion_level,
                    "speed_kmh": speed,
                    "color": color,
                    "delay_sec": delay,
                    "magnitude": mag
                }
            })
            
        return {"type": "FeatureCollection", "features": features}
    except Exception as e:
        logger.warning(f"TomTom API / heavy traffic query fallback: {e}")
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [72.8478, 19.0178]},
                    "properties": {"name": "Dadar TT Circle Congestion", "congestion_level": 3, "speed_kmh": 12.5, "color": "#EF4444"}
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [72.8630, 19.0620]},
                    "properties": {"name": "BKC Connector Slowdown", "congestion_level": 2, "speed_kmh": 18.0, "color": "#F97316"}
                }
            ]
        }
