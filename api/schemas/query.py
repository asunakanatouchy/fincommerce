"""Pydantic schemas for API request/response validation."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """Request schema for product search."""
    
    query: str = Field(..., min_length=1, max_length=500, 
                      description="User search query")
    budget: float = Field(..., gt=0, description="Maximum budget in EUR")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    category: Optional[str] = Field(None, description="Optional category filter")
    min_score: Optional[float] = Field(None, ge=0, le=1, 
                                      description="Minimum similarity threshold")
    
    @field_validator('budget')
    @classmethod
    def budget_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Budget must be positive')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "query": "laptop for development",
                "budget": 1500.0,
                "top_k": 5,
                "category": "Electronics"
            }
        }




class ProductResult(BaseModel):
    """Schema for a single product result."""
    product_id: Any
    title: str
    description: str
    price: float
    category: str
    brand: Optional[str] = None
    rating: Optional[float] = None
    semantic_score: float
    budget_fit_score: Optional[float] = None
    price_advantage_score: Optional[float] = None
    composite_score: float
    explanation: str  # User-friendly, actionable explanation
    budget_band: Optional[str] = None
    installment_available: Optional[bool] = None
    max_installments: Optional[int] = None
    shipping_days: Optional[int] = None
    msrp: Optional[float] = None
    discount_pct: Optional[float] = None



class SearchResponse(BaseModel):
    """Response schema for search results."""
    query: str
    budget: float
    total_results: int
    results: List[ProductResult]
    execution_time_ms: float
    filters_applied: Dict[str, Any]
    explanation: Optional[str] = None  # User-friendly explanation for the results
    alternatives: Optional[List[ProductResult]] = None  # Alternatives if no results
    
    class Config:
        schema_extra = {
            "example": {
                "query": "laptop for development",
                "budget": 1500.0,
                "total_results": 5,
                "results": [...],
                "execution_time_ms": 45.2,
                "filters_applied": {"budget": 1500.0},
                "explanation": "No products found for 'laptop for development' within €1500.00 budget.",
                "alternatives": [...]
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    qdrant_connected: bool
    collection_stats: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str
    detail: Optional[str] = None
    status_code: int



# Feedback schema for feedback endpoint
class FeedbackRequest(BaseModel):
    user_id: str
    action: str  # e.g., 'click', 'add_to_cart', 'purchase', 'dismiss'
    product_id: str
    query: str
    budget: float
    timestamp: float
    extra: dict = {}

__all__ = ['SearchRequest', 'SearchResponse', 'ProductResult', 
           'HealthResponse', 'ErrorResponse', 'FeedbackRequest']
