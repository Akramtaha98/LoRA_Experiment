from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="RAG Faithfulness Comparison API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompareRequest(BaseModel):
    context: str
    question: str
    reference: str

    mt5_configuration: str = "frozen"
    qwen_configuration: str = "frozen"


@app.get("/api/health")
def health():
    return {
        "status": "ok",
    }


@app.post("/api/compare")
def compare(request: CompareRequest):
    # Phase 1 mock. Replace with real model calls in later phases.
    _ = request

    mt5_result = {
        "answer": "Lyon",
        "faithfulness": 0.97,
        "exact_match": 1.0,
        "f1": 1.0,
        "latency_ms": 713,
    }

    qwen_result = {
        "answer": "Paris",
        "faithfulness": 0.31,
        "exact_match": 0.0,
        "f1": 0.0,
        "latency_ms": 482,
    }

    return {
        "mt5": mt5_result,
        "qwen": qwen_result,
    }
