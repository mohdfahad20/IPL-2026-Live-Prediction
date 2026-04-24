import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.startup import ensure_artifacts
from api.routers import predict, toss, simulation, standings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_artifacts()
    yield


app = FastAPI(
    title="IPL 2026 Predictor API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router,    prefix="/api", tags=["Prediction"])
app.include_router(toss.router,       prefix="/api", tags=["Toss"])
app.include_router(simulation.router, prefix="/api", tags=["Simulation"])
app.include_router(standings.router,  prefix="/api", tags=["Standings"])


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}