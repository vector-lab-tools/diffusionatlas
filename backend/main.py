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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# Identity handshake: every non-/health request must carry the
# `X-Diffusion-Atlas: 1` (or any non-empty value) header. Loopback
# binding already prevents off-machine access; this layer keeps any
# *other* web app on your laptop (a random localhost:9999 you happened
# to leave running) from poking /generate. The frontend adds the
# header automatically via `lib/api/client.ts`.
CLIENT_HEADER = "X-Diffusion-Atlas"

from session import session_state
from ops_generate import GenerateRequest, run as run_generate
from ops_trajectory import TrajectoryRequest, stream as stream_trajectory
from ops_score import ScoreRequest, run as run_score
from ops_warmup import WarmupRequest, run as run_warmup

app = FastAPI(title="Diffusion Atlas — Local Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    # Accept any localhost origin — http(s)://localhost or 127.0.0.1 on
    # any port. The backend binds to 127.0.0.1 only, so the trust
    # boundary is the loopback interface, not CORS. Hardcoding ports
    # was fragile: Next.js falls through to 3001/3002/… when 3000 is
    # taken by a sibling Vector Lab app (LLMbench, Manifold Atlas) and
    # any user-set `PORT=` env var landed us at "Failed to fetch".
    # Real protection comes from the X-Diffusion-Atlas handshake below.
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def identity_handshake(request: Request, call_next):
    """
    "Who are you?" → "Diffusion Atlas".

    Every state-touching endpoint requires `X-Diffusion-Atlas: <any
    non-empty value>` on the request. The header costs nothing for the
    legitimate frontend (it's added centrally by the API client) and
    means a random web app on the user's machine that doesn't know
    about the convention can't drive the local backend even if it
    finds the port.

    Exemptions:
    - `OPTIONS` — CORS preflight, no body to abuse.
    - `GET /health` — needs to be reachable before the frontend has
      bootstrapped enough to attach the header. Liveness only, no side
      effects.
    """
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.url.path == "/health" and request.method == "GET":
        return await call_next(request)
    if not request.headers.get(CLIENT_HEADER):
        return JSONResponse(
            status_code=401,
            content={
                "detail": (
                    f"Missing {CLIENT_HEADER} header. This endpoint is only "
                    "callable from the Diffusion Atlas frontend. If you're "
                    "scripting against the local backend directly, add the "
                    f"header (any non-empty value, e.g. `{CLIENT_HEADER}: 1`)."
                ),
            },
        )
    return await call_next(request)


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
