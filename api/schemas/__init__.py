"""API schemas initialization."""
from .query import (
    SearchRequest, SearchResponse, ProductResult,
    HealthResponse, ErrorResponse
)

__all__ = [
    'SearchRequest', 'SearchResponse', 'ProductResult',
    'HealthResponse', 'ErrorResponse'
]
