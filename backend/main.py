"""
Diffusion Atlas — Local FastAPI Backend

Run from this directory:
    uvicorn main:app --reload --port 8000

Listens on http://localhost:8000 with CORS open to localhost:3000 (the
Next.js frontend). Single-tenant by design — one user, one machine.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from session import session_state
from ops_generate import GenerateRequest, run as run_generate

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
