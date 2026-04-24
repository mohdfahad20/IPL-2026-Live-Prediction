from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services.prediction_service import predict_match

router = APIRouter()


class PredictRequest(BaseModel):
    team1: str
    team2: str
    season: str = "2026"
    venue: str | None = None
    toss_winner: str | None = None
    toss_decision: str | None = None


@router.post("/predict")
def predict(req: PredictRequest):
    try:
        return predict_match(
            team1=req.team1,
            team2=req.team2,
            season=req.season,
            venue=req.venue,
            toss_winner=req.toss_winner,
            toss_decision=req.toss_decision,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))