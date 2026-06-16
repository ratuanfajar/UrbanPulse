"""FastAPI route definitions."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from chatbot.schemas import ChatRequest, ChatResponse, HealthResponse, Source
from chatbot.rag_service import RAGService, SearchError

logger = logging.getLogger(__name__)
router = APIRouter()


def get_rag_service(request: Request) -> RAGService:
    """Ambil RAGService dari app.state (di-set di lifespan handler app.py)."""
    return request.app.state.rag_service


@router.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health(rag: RAGService = Depends(get_rag_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_ready=rag.llm_ready,
        search_ready=rag.search_ready,
    )


@router.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    body: ChatRequest,
    rag: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    """Answer a government user's question using web-search-grounded RAG."""
    try:
        result = await rag.ask(body.question, session_id=body.session_id)
    except SearchError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during RAG inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Check the server logs.",
        )

    return ChatResponse(
        question=body.question,
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
    )