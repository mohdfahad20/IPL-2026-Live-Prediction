from fastapi import APIRouter, HTTPException
from api.services.simulation_service import (
    get_latest_probabilities,
    get_probability_history,
)

router = APIRouter()


@router.get("/probabilities")
def probabilities():
    try:
        data = get_latest_probabilities()
        if not data:
            raise HTTPException(
                status_code=404,
                detail="No simulation results found. Run simulate.py first."
            )
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/probabilities/history")
def probabilities_history():
    try:
        return get_probability_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))