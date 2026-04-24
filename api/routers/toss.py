from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services.toss_service import toss_prediction

router = APIRouter()


class TossRequest(BaseModel):
    team1: str
    team2: str
    season: str = "2026"
    venue: str | None = None
    toss_winner: str
    toss_decision: str


@router.post("/toss")
def toss(req: TossRequest):
    try:
        return toss_prediction(
            team1=req.team1,
            team2=req.team2,
            toss_winner=req.toss_winner,
            toss_decision=req.toss_decision,
            season=req.season,
            venue=req.venue,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))