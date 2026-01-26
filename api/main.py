"""Main FastAPI application for FinCommerce Vector Search."""
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core import CONFIG
from src.retrieval.embedder import EmbeddingService
from src.retrieval.search_engine import VectorDB
from src.processing.ranker import Ranker
from api.schemas import SearchRequest, SearchResponse, HealthResponse, ProductResult

# Configure logging
logging.basicConfig(
    level=CONFIG.get('logging', {}).get('level', 'INFO'),
    format=CONFIG.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger = logging.getLogger(__name__)

# Global service instances
embedding_service: EmbeddingService = None
vector_db: VectorDB = None
ranker: Ranker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global embedding_service, vector_db, ranker
    
    # STARTUP
    logger.info("🚀 Starting FinCommerce Vector Search API...")
    
    try:
        # Initialize embedding service
        embedding_config = CONFIG.get('embeddings', {})
        embedding_service = EmbeddingService(
            model_name=embedding_config.get('model_name', 'all-MiniLM-L6-v2')
        )
        logger.info("✓ Embedding service initialized")
        
        # Initialize vector database
        qdrant_config = CONFIG.get('qdrant', {})
        vector_db = VectorDB(
            host=qdrant_config.get('host', 'localhost'),
            port=qdrant_config.get('port', 6333),
            collection_name=qdrant_config.get('collection_name', 'products'),
            vector_size=qdrant_config.get('vector_size', 384)
        )
        logger.info("✓ Vector database connected")
        
        # Initialize ranker
        search_config = CONFIG.get('search', {})
        from src.processing.ranker import ScoringWeights
        weights = ScoringWeights(
            semantic=search_config.get('semantic_weight', 0.6),
            budget_fit=search_config.get('budget_weight', 0.3),
            price_advantage=search_config.get('price_advantage_weight', 0.1)
        )
        ranker = Ranker(weights=weights)
        logger.info("✓ Ranker initialized")
        
        logger.info("✅ All systems ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield  # Application runs
    
    # SHUTDOWN
    logger.info("Shutting down...")


# Create FastAPI app
api_config = CONFIG.get('api', {})
app = FastAPI(
    title=api_config.get('title', 'FinCommerce Vector Search'),
    description="Context-aware product search with Qdrant vector database",
    version=api_config.get('version', '0.1.0'),
    lifespan=lifespan
)

# Add CORS middleware
cors_origins = api_config.get('cors_origins', ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": "FinCommerce Vector Search",
        "version": api_config.get('version', '0.1.0'),
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    if vector_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Services not initialized"
        )
    
    try:
        qdrant_healthy = vector_db.health_check()
        stats = vector_db.get_stats() if qdrant_healthy else None
        
        return HealthResponse(
            status="healthy" if qdrant_healthy else "degraded",
            qdrant_connected=qdrant_healthy,
            collection_stats=stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_products(request: SearchRequest):
    """Search for products with semantic similarity and budget constraints.
    
    This endpoint performs:
    1. Query embedding using SentenceTransformers
    2. Semantic search in Qdrant vector database
    3. Budget-aware filtering
    4. Composite scoring and ranking
    5. Explainable recommendations
    """
    start_time = time.time()
    
    try:
        # Validate services
        if not all([embedding_service, vector_db, ranker]):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Services not fully initialized"
            )
        
        # Generate query embedding
        query_vector = embedding_service.embed(request.query)
        
        # Search with constraints
        search_results = vector_db.search(
            query_vector=query_vector,
            max_budget=request.budget,
            category=request.category,
            top_k=request.top_k,
            score_threshold=request.min_score
        )
        
        # Rank and explain results
        ranked_results = ranker.rank(
            search_hits=search_results,
            user_budget=request.budget,
            min_score=request.min_score or 0.0
        )
        
        # Convert to response format
        product_results = [ProductResult(**result) for result in ranked_results]
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        filters_applied = {
            "budget": request.budget,
            "category": request.category,
            "min_score": request.min_score
        }
        
        return SearchResponse(
            query=request.query,
            budget=request.budget,
            total_results=len(product_results),
            results=product_results,
            execution_time_ms=round(execution_time, 2),
            filters_applied=filters_applied
        )
        
    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search operation failed"
        )


@app.get("/stats", tags=["Stats"])
async def get_stats():
    """Get system statistics."""
    if vector_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database not initialized"
        )
    
    try:
        stats = vector_db.get_stats()
        return {
            **stats,
            "embedding_model": embedding_service.model_name if embedding_service else None,
            "vector_dimension": embedding_service.vector_size if embedding_service else None
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=api_config.get('reload', True),
        log_level=CONFIG.get('logging', {}).get('level', 'info').lower()
    )
