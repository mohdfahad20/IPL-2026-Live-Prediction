from fastapi import APIRouter, HTTPException
from api.services.standings_service import (
    get_standings,
    get_recent_matches,
    get_venues,
)

router = APIRouter()


@router.get("/standings")
def standings():
    try:
        return {"standings": get_standings()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent-matches")
def recent_matches():
    try:
        return get_recent_matches()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/venues")
def venues():
    try:
        return get_venues()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))