"""
Diffusion Atlas — Local FastAPI Backend

Run from this directory:
    uvicorn main:app --reload --port 8000

Listens on http://localhost:8000 with CORS open to localhost:3000 (the
Next.js frontend). Single-tenant by design — one user, one machine.
"""
from __future__ import annotations

import os

# Tell PyTorch to release MPS memory back to the system below these
# fractions of total RAM. On a 24 GB MacBook running SD 1.5 at fp32 plus
# OS + Chrome + Claude Code, the default (1.0) regularly pushes the
# kernel into encrypted swap and freezes the machine. 0.7 leaves ~7 GB
# of headroom for everything else.
#
# Why both: when only HIGH is set, PyTorch derives `LOW = HIGH * 2`, so
# 0.7 produces an invalid low of 1.4 ("must be ≤ 1.0"). Setting both
# explicitly to 0.7 / 0.5 means the allocator starts soft-evicting at
# 50% RAM and hard-caps at 70%. Set BEFORE torch is imported anywhere
# (FastAPI, session, ops_*) or it has no effect.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from session import session_state
from ops_generate import GenerateRequest, run as run_generate
from ops_trajectory import TrajectoryRequest, stream as stream_trajectory
from ops_score import ScoreRequest, run as run_score
from ops_warmup import WarmupRequest, run as run_warmup

app = FastAPI(title="Diffusion Atlas — Local Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return session_state.health()


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    return run_generate(req, session_state)


@app.post("/trajectory")
def trajectory(req: TrajectoryRequest):
    return stream_trajectory(req, session_state)


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    return run_score(req, session_state)


@app.post("/warmup")
def warmup(req: WarmupRequest) -> dict:
    return run_warmup(req, session_state)
