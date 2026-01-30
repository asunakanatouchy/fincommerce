import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Main FastAPI application for FinCommerce Vector Search.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.core import CONFIG
from src.retrieval.embedder import EmbeddingService
from src.retrieval.search_engine import VectorDB
from src.processing.ranker import Ranker, Explainer
from api.schemas import SearchRequest, SearchResponse, HealthResponse, ProductResult
from src.processing.ranker import ScoringWeights

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=CONFIG.get("logging", {}).get("level", "INFO"),
    format=CONFIG.get("logging", {}).get(
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ),
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------------------
embedding_service: EmbeddingService = None
vector_db: VectorDB = None
ranker: Ranker = None

user_feedback_store = []  # in-memory (demo / hackathon)

# ------------------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_service, vector_db, ranker

    logger.info("🚀 Starting FinCommerce Vector Search API...")
    start_time = time.time()

    try:
        # Embeddings
        embedding_config = CONFIG.get("embeddings", {})
        embedding_service = EmbeddingService(
            model_name=embedding_config.get(
                "model_name", "all-MiniLM-L6-v2"
            )
        )
        logger.info("✓ Embedding service initialized")

        # Vector DB
        qdrant_config = CONFIG.get("qdrant", {})
        vector_db = VectorDB(
            host=qdrant_config.get("host", "localhost"),
            port=qdrant_config.get("port", 6333),
            collection_name=qdrant_config.get(
                "collection_name", "products"
            ),
            vector_size=qdrant_config.get("vector_size", 384),
        )
        logger.info("✓ Vector database connected")

        # Ranker
        search_config = CONFIG.get("search", {})
        weights = ScoringWeights(
            semantic=search_config.get("semantic_weight", 0.6),
            budget_fit=search_config.get("budget_weight", 0.3),
            price_advantage=search_config.get(
                "price_advantage_weight", 0.1
            ),
        )
        ranker = Ranker(weights=weights)
        logger.info("✓ Ranker initialized")

        logger.info("✅ All systems ready!")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    yield

    logger.info("Shutting down...")

# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------
api_config = CONFIG.get("api", {})

app = FastAPI(
    title=api_config.get("title", "FinCommerce Vector Search"),
    description="Context-aware product search with Qdrant vector database",
    version=api_config.get("version", "0.1.0"),
    lifespan=lifespan,
)

# ------------------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# System stats endpoint (FIXED POSITION)
# ------------------------------------------------------------------------------
@app.get("/stats", tags=["System"])
async def stats():
    if vector_db is None:
        return {
            "points_count": 0,
            "vector_size": 0,
            "embedding_model": None,
        }

    try:
        stats = vector_db.get_stats() or {}

        if embedding_service:
            stats["embedding_model"] = getattr(
                embedding_service, "model_name", None
            )

        return {
            "points_count": stats.get("points_count", 0),
            "vector_size": stats.get("vector_size", 0),
            "embedding_model": stats.get("embedding_model", None),
        }

    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        return {
            "points_count": 0,
            "vector_size": 0,
            "embedding_model": None,
        }

# ------------------------------------------------------------------------------
# Feedback
# ------------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    user_id: str
    action: str
    product_id: str
    query: str
    budget: float
    timestamp: float
    extra: dict = {}

@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(feedback: FeedbackRequest):
    user_feedback_store.append(feedback.model_dump())
    return {"status": "ok", "message": "Feedback received"}

# ------------------------------------------------------------------------------
# Root & Health
# ------------------------------------------------------------------------------
@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "FinCommerce Vector Search",
        "version": api_config.get("version", "0.1.0"),
        "status": "online",
        "docs": "/docs",
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    if vector_db is None:
        return HealthResponse(
            status="degraded",
            qdrant_connected=False,
            collection_stats=None,
        )

    try:
        healthy = vector_db.health_check()
        stats = vector_db.get_stats() if healthy else None
        return HealthResponse(
            status="healthy" if healthy else "degraded",
            qdrant_connected=healthy,
            collection_stats=stats,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            qdrant_connected=False,
            collection_stats=None,
        )

# ------------------------------------------------------------------------------
# Search
# ------------------------------------------------------------------------------
@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_products(request: SearchRequest):
    if not all([embedding_service, vector_db, ranker]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Services not fully initialized",
        )

    query_vector = embedding_service.embed(request.query)
    start_time = time.time()

    try:
        hits = vector_db.search(
            query_vector=query_vector,
            max_budget=request.budget,
            category=request.category,
            top_k=request.top_k,
            score_threshold=request.min_score,
            query_text=request.query,
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        hits = []

    ranked = ranker.rank(
        search_hits=hits,
        user_budget=request.budget,
        min_score=request.min_score or 0.0,
    )

    results = [ProductResult(**r) for r in ranked]
    exec_ms = max((time.time() - start_time) * 1000, 0.01)

    return {
        "query": request.query,
        "budget": request.budget,
        "total_results": len(results),
        "results": results,
        "execution_time_ms": round(exec_ms, 2),
        "filters_applied": {
            "budget": request.budget,
            "category": request.category,
            "min_score": request.min_score,
        },
        "explanation": None,
        "similar_items": None,
    }

# ------------------------------------------------------------------------------
# Global exception handler
# ------------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=api_config.get("reload", True),
        log_level=CONFIG.get("logging", {})
        .get("level", "info")
        .lower(),
    )
