import logging
from api.services.prediction_service import predict_match

log = logging.getLogger(__name__)


def toss_prediction(
    team1: str,
    team2: str,
    toss_winner: str,
    toss_decision: str,
    season: str = "2026",
    venue: str | None = None,
) -> dict:
    """
    Runs prediction twice — before and after toss —
    and returns the comparison with shift + impact label.
    """
    pre = predict_match(
        team1=team1,
        team2=team2,
        season=season,
        venue=venue,
        toss_winner=None,
        toss_decision=None,
    )

    post = predict_match(
        team1=team1,
        team2=team2,
        season=season,
        venue=venue,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
    )

    shift = round(post["p_team1_wins"] - pre["p_team1_wins"], 4)

    return {
        "team1":         team1,
        "team2":         team2,
        "toss_winner":   toss_winner,
        "toss_decision": toss_decision,
        "venue":         venue,
        "pre_toss":      pre,
        "post_toss":     post,
        "toss_shift":    shift,
        "impact":        _impact_label(shift),
        "beneficiary":   team1 if shift > 0 else team2,
    }


def _impact_label(shift: float) -> str:
    abs_shift = abs(shift)
    if abs_shift > 0.05:
        return "High"
    if abs_shift > 0.02:
        return "Moderate"
    return "Low"