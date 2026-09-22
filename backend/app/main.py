from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Baymax API",
    description="AI healthcare companion backend (RAG-grounded).",
    version="0.1.0",
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
    """Stub for now — not wired to the RAG pipeline yet."""
    return ChatResponse(
        reply=f"(stub) I heard: '{request.message}'. RAG pipeline not connected yet.",
        sources=[],
        urgency="unknown",
    )