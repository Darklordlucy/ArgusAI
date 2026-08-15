import datetime
import logging
from typing import Optional, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.config import get_db
from app.models.db_models import SOSAlert, IoTReading

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/iot", tags=["IoT"])

# Flexible Pydantic Request Schemas for IoT hardware
class SOSAlertCreate(BaseModel):
    device_id: Optional[str] = Field(default="IOT-DEVICE-01", alias="deviceId")
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    lng: Optional[float] = None
    triggered_at: Optional[datetime.datetime] = None
    resolved: Optional[bool] = False
    hospital_notified: Optional[bool] = True

    class Config:
        populate_by_name = True

class TelemetryCreate(BaseModel):
    device_id: Optional[str] = "SENSOR-01"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    accel_x: Optional[float] = 0.0
    accel_y: Optional[float] = 0.0
    accel_z: Optional[float] = 9.81
    gyro_x: Optional[float] = 0.0
    gyro_y: Optional[float] = 0.0
    gyro_z: Optional[float] = 0.0
    speed_kmh: Optional[float] = 0.0

    class Config:
        populate_by_name = True


@router.get("")
async def iot_placeholder():
    return {
        "status": "online",
        "service": "IoT Ingestion Router",
        "endpoints": ["POST /api/v1/iot/sos", "POST /api/v1/iot/telemetry"]
    }


# GET /api/v1/iot/sos
@router.get("/sos")
async def get_sos_alerts(db: AsyncSession = Depends(get_db)):
    """Fetch recent active SOS alerts or return operational guide."""
    try:
        result = await db.execute(select(SOSAlert).order_by(SOSAlert.triggered_at.desc()).limit(50))
        alerts = result.scalars().all()
        return {
            "status": "ok",
            "message": "Send an HTTP POST request to /api/v1/iot/sos with JSON body to record crash/SOS alerts.",
            "count": len(alerts),
            "alerts": [
                {
                    "id": a.id,
                    "device_id": a.device_id,
                    "latitude": a.latitude,
                    "longitude": a.longitude,
                    "triggered_at": a.triggered_at.isoformat() if a.triggered_at else None,
                    "resolved": a.resolved,
                    "hospital_notified": a.hospital_notified
                }
                for a in alerts
            ]
        }
    except Exception as e:
        logger.warning(f"DB read exception in GET /sos: {e}")
        return {
            "status": "ok",
            "message": "Send an HTTP POST request to /api/v1/iot/sos with JSON body to record crash/SOS alerts.",
            "db_status": f"disconnected or idle ({str(e)})"
        }


# POST /api/v1/iot/sos (Inserts into sos_alerts table)
@router.post("/sos", status_code=status.HTTP_201_CREATED)
async def create_sos_alert(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Robust IoT Crash / Emergency SOS Ingestion Endpoint.
    Parses any JSON format from ESP32 / Arduino / Microcontrollers / Mobile Apps.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Extract coordinates flexibly (supports latitude/longitude or lat/lon/lng)
    latitude = body.get("latitude") or body.get("lat") or body.get("lat_deg")
    longitude = body.get("longitude") or body.get("lon") or body.get("lng") or body.get("lon_deg")
    device_id = str(body.get("device_id") or body.get("deviceId") or body.get("device") or body.get("id") or "IOT-DEVICE-CRASH")

    if latitude is None or longitude is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing coordinate parameters. Please provide 'latitude' and 'longitude' (or 'lat' and 'lon')."
        )

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Coordinates 'latitude' and 'longitude' must be valid numeric floating-point numbers."
        )

    hospital_notified = bool(body.get("hospital_notified", True))
    resolved = bool(body.get("resolved", False))
    now = datetime.datetime.utcnow()

    try:
        alert = SOSAlert(
            device_id=device_id,
            latitude=latitude,
            longitude=longitude,
            triggered_at=now,
            resolved=resolved,
            hospital_notified=hospital_notified
        )
        db.add(alert)
        await db.commit()
        await db.refresh(alert)

        logger.info(f"Successfully recorded IoT SOS Alert: Device {device_id} at ({latitude}, {longitude})")

        return {
            "status": "success",
            "message": "Emergency SOS alert recorded in database successfully.",
            "alert": {
                "id": alert.id,
                "device_id": alert.device_id,
                "latitude": alert.latitude,
                "longitude": alert.longitude,
                "triggered_at": alert.triggered_at.isoformat(),
                "resolved": alert.resolved,
                "hospital_notified": alert.hospital_notified
            }
        }
    except Exception as e:
        await db.rollback()
        logger.error(f"Database insertion failure in POST /sos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database insertion failed: {str(e)}"
        )


# POST /api/v1/iot/telemetry
@router.post("/telemetry", status_code=status.HTTP_201_CREATED)
async def upload_telemetry(request: Request, db: AsyncSession = Depends(get_db)):
    """Record IoT sensor telemetry data into database."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    latitude = body.get("latitude") or body.get("lat")
    longitude = body.get("longitude") or body.get("lon")
    device_id = str(body.get("device_id") or body.get("deviceId") or "SENSOR-DEVICE")

    if latitude is None or longitude is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing coordinate parameters 'latitude' and 'longitude'."
        )

    try:
        reading = IoTReading(
            device_id=device_id,
            latitude=float(latitude),
            longitude=float(longitude),
            accel_x=float(body.get("accel_x", 0.0)),
            accel_y=float(body.get("accel_y", 0.0)),
            accel_z=float(body.get("accel_z", 9.81)),
            gyro_x=float(body.get("gyro_x", 0.0)),
            gyro_y=float(body.get("gyro_y", 0.0)),
            gyro_z=float(body.get("gyro_z", 0.0)),
            timestamp=datetime.datetime.utcnow()
        )
        db.add(reading)
        await db.commit()
        await db.refresh(reading)
        return {"status": "success", "id": reading.id}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record telemetry: {str(e)}"
        )
