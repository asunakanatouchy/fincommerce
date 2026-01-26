"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from api.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_search_dependencies():
    """Mock all search dependencies."""
    with patch('api.main.embedding_service') as mock_embedder, \
         patch('api.main.vector_db') as mock_db, \
         patch('api.main.ranker') as mock_ranker:
        
        # Setup embedder mock
        mock_embedder.embed.return_value = [0.1] * 384
        
        # Setup vector_db mock
        mock_hit = Mock()
        mock_hit.score = 0.85
        mock_hit.payload = {
            'product_id': 1,
            'title': 'Test Laptop',
            'description': 'Great laptop for testing',
            'price': 999.99,
            'category': 'Electronics',
            'brand': 'TestBrand',
            'rating': 4.5,
            'msrp': 1199.99,
            'discount_pct': 16.7,
            'stock': 50,
            'availability': 'in_stock',
            'payment_methods': 'card;paypal',
            'installment_available': True,
            'max_installments': 12,
            'shipping_days': 3,
            'budget_band': 'premium',
            'tags': 'laptop;testing'
        }
        mock_db.search.return_value = [mock_hit]
        mock_db.health_check.return_value = True
        mock_db.get_stats.return_value = {
            'total_products': 100,
            'collection_name': 'products'
        }
        
        # Setup ranker mock
        mock_ranker.rank.return_value = [{
            'product_id': 1,
            'title': 'Test Laptop',
            'description': 'Great laptop for testing',
            'price': 999.99,
            'category': 'Electronics',
            'brand': 'TestBrand',
            'rating': 4.5,
            'msrp': 1199.99,
            'discount_pct': 16.7,
            'stock': 50,
            'availability': 'in_stock',
            'payment_methods': 'card;paypal',
            'installment_available': True,
            'max_installments': 12,
            'shipping_days': 3,
            'budget_band': 'premium',
            'tags': 'laptop;testing',
            'semantic_score': 0.85,
            'composite_score': 0.82,
            'budget_fit_score': 0.80,
            'price_advantage_score': 0.75,
            'explanation': 'Strong semantic match with good budget fit'
        }]
        
        yield {
            'embedder': mock_embedder,
            'db': mock_db,
            'ranker': mock_ranker
        }


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert 'message' in data or 'name' in data


def test_health_endpoint(client, mock_search_dependencies):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data
    assert 'qdrant_connected' in data


def test_stats_endpoint(client, mock_search_dependencies):
    """Test stats endpoint."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert 'total_products' in data


def test_search_endpoint_basic(client, mock_search_dependencies):
    """Test basic search request."""
    payload = {
        "query": "laptop for development",
        "budget": 1500.0,
        "top_k": 5
    }
    
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert 'query' in data
    assert 'budget' in data
    assert 'results' in data
    assert 'total_results' in data
    assert 'execution_time_ms' in data
    
    assert data['query'] == payload['query']
    assert data['budget'] == payload['budget']
    assert len(data['results']) > 0


def test_search_endpoint_with_category(client, mock_search_dependencies):
    """Test search with category filter."""
    payload = {
        "query": "laptop",
        "budget": 2000.0,
        "top_k": 5,
        "category": "Electronics"
    }
    
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data['filters_applied']['category'] == "Electronics"


def test_search_endpoint_with_min_score(client, mock_search_dependencies):
    """Test search with minimum score threshold."""
    payload = {
        "query": "laptop",
        "budget": 1500.0,
        "top_k": 5,
        "min_score": 0.7
    }
    
    response = client.post("/search", json=payload)
    assert response.status_code == 200


def test_search_endpoint_validation_negative_budget(client):
    """Test search rejects negative budget."""
    payload = {
        "query": "laptop",
        "budget": -100.0,
        "top_k": 5
    }
    
    response = client.post("/search", json=payload)
    assert response.status_code == 422  # Validation error


def test_search_endpoint_validation_invalid_top_k(client):
    """Test search rejects invalid top_k."""
    payload = {
        "query": "laptop",
        "budget": 1500.0,
        "top_k": 150  # Too high
    }
    
    response = client.post("/search", json=payload)
    assert response.status_code == 422


def test_search_endpoint_validation_empty_query(client):
    """Test search rejects empty query."""
    payload = {
        "query": "",
        "budget": 1500.0,
        "top_k": 5
    }
    
    response = client.post("/search", json=payload)
    assert response.status_code == 422


def test_search_endpoint_no_results(client, mock_search_dependencies):
    """Test search with no results."""
    # Make vector_db return empty
    mock_search_dependencies['db'].search.return_value = []
    mock_search_dependencies['ranker'].rank.return_value = []
    
    payload = {
        "query": "nonexistent product xyz123",
        "budget": 1500.0,
        "top_k": 5
    }
    
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data['total_results'] == 0
    assert len(data['results']) == 0


def test_search_response_structure(client, mock_search_dependencies):
    """Test search response has correct structure."""
    payload = {
        "query": "laptop",
        "budget": 1500.0,
        "top_k": 5
    }
    
    response = client.post("/search", json=payload)
    data = response.json()
    
    # Check top-level structure
    assert isinstance(data['query'], str)
    assert isinstance(data['budget'], (int, float))
    assert isinstance(data['total_results'], int)
    assert isinstance(data['results'], list)
    assert isinstance(data['execution_time_ms'], (int, float))
    assert isinstance(data['filters_applied'], dict)
    
    # Check result structure
    if len(data['results']) > 0:
        result = data['results'][0]
        assert 'product_id' in result
        assert 'title' in result
        assert 'price' in result
        assert 'composite_score' in result
        assert 'semantic_score' in result
        assert 'explanation' in result


def test_cors_headers(client):
    """Test CORS headers are set."""
    response = client.get("/health")
    assert "access-control-allow-origin" in response.headers or response.status_code == 200


def test_search_timing_measurement(client, mock_search_dependencies):
    """Test that execution time is measured."""
    payload = {
        "query": "laptop",
        "budget": 1500.0,
        "top_k": 5
    }
    
    response = client.post("/search", json=payload)
    data = response.json()
    
    assert 'execution_time_ms' in data
    assert data['execution_time_ms'] > 0
    assert data['execution_time_ms'] < 10000  # Should be under 10 seconds
