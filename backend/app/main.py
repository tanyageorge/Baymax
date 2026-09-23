from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.llm_client import generate_answer
from app.retrieval import build_index, retrieve_chunks

# Built once at server startup, reused for every /chat request.
_index = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index
    print("Building retrieval index...")
    _index = build_index()
    print("Retrieval index ready.")
    yield
    # (nothing needed on shutdown for now)


app = FastAPI(
    title="Baymax API",
    description="AI healthcare companion backend (RAG-grounded).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Liveness check — hit this to confirm the server is running."""
    return {"status": "ok", "service": "baymax-backend"}


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = []
    urgency: str = "unknown"


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    retrieved_chunks = retrieve_chunks(_index, request.message)
    answer = generate_answer(request.message, retrieved_chunks)
    sources = sorted({chunk["source"] for chunk in retrieved_chunks})

    return ChatResponse(
        reply=answer,
        sources=sources,
        # Real urgency classification is a later milestone (the Triage
        # Agent) — this endpoint currently only does retrieve + generate.
        urgency="unknown",
    )