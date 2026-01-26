"""Test configuration and fixtures."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def test_data_dir():
    """Return test data directory path."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_products():
    """Return sample products for testing with full metadata."""
    return [
        {
            "id": 1,
            "title": "Dev Laptop Pro",
            "description": "High-performance laptop for software development with 32GB RAM",
            "price": 1299.99,
            "category": "Electronics",
            "brand": "TechBrand",
            "rating": 4.7,
            "msrp": 1499.99,
            "discount_pct": 13.3,
            "stock": 50,
            "availability": "in_stock",
            "payment_methods": "card;paypal;apple_pay",
            "installment_available": True,
            "max_installments": 12,
            "shipping_days": 3,
            "budget_band": "premium",
            "tags": "laptop;development;premium"
        },
        {
            "id": 2,
            "title": "Budget Laptop Basic",
            "description": "Affordable laptop for everyday use and light coding",
            "price": 499.99,
            "category": "Electronics",
            "brand": "ValueBrand",
            "rating": 3.9,
            "msrp": 599.99,
            "discount_pct": 16.7,
            "stock": 100,
            "availability": "in_stock",
            "payment_methods": "card;paypal",
            "installment_available": False,
            "max_installments": 0,
            "shipping_days": 5,
            "budget_band": "budget",
            "tags": "laptop;budget;student"
        },
        {
            "id": 3,
            "title": "Ergonomic Mouse Pro",
            "description": "Wireless ergonomic mouse for developers",
            "price": 79.99,
            "category": "Electronics",
            "brand": "ErgoTech",
            "rating": 4.5,
            "msrp": 99.99,
            "discount_pct": 20.0,
            "stock": 200,
            "availability": "in_stock",
            "payment_methods": "card;paypal",
            "installment_available": False,
            "max_installments": 0,
            "shipping_days": 2,
            "budget_band": "budget",
            "tags": "mouse;ergonomic;wireless"
        }
    ]


@pytest.fixture
def sample_query():
    """Sample search query."""
    return "laptop for development"


@pytest.fixture
def sample_budget():
    """Sample budget constraint."""
    return 1500.0


@pytest.fixture
def mock_search_hit():
    """Mock Qdrant search hit object."""
    class MockHit:
        def __init__(self, score, payload):
            self.score = score
            self.payload = payload
    
    return MockHit(
        score=0.85,
        payload={
            'product_id': 1,
            'title': 'Test Laptop Pro',
            'description': 'High-performance development laptop',
            'price': 1200.0,
            'category': 'Electronics',
            'brand': 'TestBrand',
            'rating': 4.5,
            'msrp': 1400.0,
            'discount_pct': 14.3,
            'stock': 50,
            'availability': 'in_stock',
            'payment_methods': 'card;paypal',
            'installment_available': True,
            'max_installments': 12,
            'shipping_days': 3,
            'budget_band': 'premium',
            'tags': 'laptop;development'
        }
    )


@pytest.fixture
def sample_embedding():
    """Sample 384-dimensional embedding vector."""
    import numpy as np
    return np.random.randn(384).tolist()
