from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_db
from app.schemas.route_schemas import RouteRequest, FeedbackRequest
from app.services.route_service import RouteService

router = APIRouter(prefix="/api/v1/routes", tags=["Routes"])
route_service = RouteService()

@router.get("/types")
async def get_route_types():
    """Retrieve all supported routing objective weight functions."""
    return {
        "types": [
            {"id": "fastest", "name": "Fastest Route", "description": "Minimizes driving time using traffic metrics.", "icon": "Zap"},
            {"id": "safest", "name": "Safest Route", "description": "Avoids segment hazards and storm weather.", "icon": "Shield"},
            {"id": "straightest", "name": "Straightest Route", "description": "Minimizes turns and angular bearing shifts.", "icon": "ArrowRight"},
            {"id": "popular", "name": "Popular Route", "description": "Scenic routing traversing points of interest.", "icon": "Star"}
        ]
    }

@router.post("/compute")
async def compute_route(
    request: RouteRequest,
    db: AsyncSession = Depends(get_db)
):
    """Snaps origin/destination and runs the dynamic route optimization engine."""
    try:
        result = await route_service.compute_route_service(
            db=db,
            origin={"lat": request.origin.lat, "lon": request.origin.lon},
            destination={"lat": request.destination.lat, "lon": request.destination.lon},
            route_type=request.route_type,
            vehicle_type=request.vehicle_type,
            avoid_tolls=request.avoid_tolls
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Routing failure: {str(e)}"
        )

@router.get("/hazards")
async def get_hazards_heatmap(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve all road segment hazard points within a bounding box."""
    try:
        query = text("""
            SELECT 
                id,
                latitude,
                longitude,
                hazard_score,
                hazard_type
            FROM segment_hazards
            WHERE latitude BETWEEN :min_lat AND :max_lat 
              AND longitude BETWEEN :min_lon AND :max_lon
            ORDER BY recorded_at DESC
            LIMIT 5000
        """)
        
        result = await db.execute(query, {
            "min_lat": min_lat,
            "min_lon": min_lon,
            "max_lat": max_lat,
            "max_lon": max_lon
        })
        rows = result.fetchall()
        
        hazards_list = []
        for hid, lat, lon, hazard_score, hazard_type in rows:
            if lat is None or lon is None:
                continue

            hazards_list.append({
                "id": hid,
                "latitude": float(lat),
                "longitude": float(lon),
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)]
                },
                "hazard_score": float(hazard_score or 0.0),
                "hazard_type": hazard_type or "unknown"
            })
            
        return {"hazards": hazards_list}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve hazard heatmap: {str(e)}"
        )

@router.post("/feedback")
async def log_route_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db)
):
    """Submit RLHF routing feedback directly into database with robust PostGIS and standard fallback."""
    try:
        coords_str = ", ".join([f"{c[0]} {c[1]}" for c in request.route_geometry]) if request.route_geometry else f"{request.start_point.lon} {request.start_point.lat}, {request.end_point.lon} {request.end_point.lat}"
        line_wkt = f"LINESTRING({coords_str})"

        query = text("""
            INSERT INTO route_feedback (
                user_id, start_point, end_point, route_geometry, route_type, rating, feedback_text, created_at
            ) VALUES (
                :user_id,
                ST_SetSRID(ST_MakePoint(:start_lon, :start_lat), 4326),
                ST_SetSRID(ST_MakePoint(:end_lon, :end_lat), 4326),
                ST_GeomFromText(:line_wkt, 4326),
                :route_type,
                :rating,
                :feedback_text,
                NOW()
            );
        """)
        
        await db.execute(query, {
            "user_id": request.user_id or "pilot_driver",
            "start_lon": float(request.start_point.lon),
            "start_lat": float(request.start_point.lat),
            "end_lon": float(request.end_point.lon),
            "end_lat": float(request.end_point.lat),
            "line_wkt": line_wkt,
            "route_type": request.route_type or "fastest",
            "rating": int(request.rating),
            "feedback_text": request.feedback_text or ""
        })
        await db.commit()
        return {"status": "success", "message": "Feedback inserted into database successfully."}
    except Exception as e:
        await db.rollback()
        print(f"[log_route_feedback] Primary PostGIS insert warning: {e}. Executing standard fallback query...")
        try:
            fallback_query = text("""
                INSERT INTO route_feedback (
                    user_id, route_type, rating, feedback_text, created_at
                ) VALUES (
                    :user_id, :route_type, :rating, :feedback_text, NOW()
                );
            """)
            await db.execute(fallback_query, {
                "user_id": request.user_id or "pilot_driver",
                "route_type": request.route_type or "fastest",
                "rating": int(request.rating),
                "feedback_text": request.feedback_text or ""
            })
            await db.commit()
            return {"status": "success", "message": "Feedback inserted into database via standard query."}
        except Exception as fallback_err:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to insert feedback into database: {str(fallback_err)}"
            )
