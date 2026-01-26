"""Tests for vector database operations."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.retrieval.search_engine import VectorDB, FinSearchEngine


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    with patch('src.retrieval.search_engine.QdrantClient') as mock:
        client = Mock()
        mock.return_value = client
        yield client


def test_vector_db_initialization(mock_qdrant_client):
    """Test VectorDB initializes correctly."""
    db = VectorDB(host="localhost", port=6333, collection_name="test_products")
    assert db.collection_name == "test_products"
    assert db.vector_size == 384


def test_create_collection(mock_qdrant_client):
    """Test collection creation."""
    mock_qdrant_client.collection_exists.return_value = False
    
    db = VectorDB(host="localhost", port=6333)
    db.create_collection()
    
    # Verify collection was created
    assert mock_qdrant_client.create_collection.called


def test_create_collection_already_exists(mock_qdrant_client):
    """Test creating collection when it already exists."""
    mock_qdrant_client.collection_exists.return_value = True
    
    db = VectorDB(host="localhost", port=6333)
    db.create_collection()
    
    # Should recreate collection
    assert mock_qdrant_client.delete_collection.called
    assert mock_qdrant_client.create_collection.called


def test_index_products_batching(mock_qdrant_client, sample_products):
    """Test product indexing with batching."""
    db = VectorDB(host="localhost", port=6333, batch_size=2)
    
    # Mock embeddings
    embeddings = [[0.1] * 384 for _ in sample_products]
    
    db.index_products(sample_products, embeddings)
    
    # Should be called due to batching (3 products, batch_size=2)
    assert mock_qdrant_client.upsert.call_count >= 1


def test_search_basic(mock_qdrant_client, sample_embedding):
    """Test basic search functionality."""
    mock_result = Mock()
    mock_result.id = 1
    mock_result.score = 0.85
    mock_result.payload = {'title': 'Test Product', 'price': 100.0}
    
    mock_qdrant_client.search.return_value = [mock_result]
    
    db = VectorDB(host="localhost", port=6333)
    results = db.search(sample_embedding, top_k=5)
    
    assert len(results) == 1
    assert results[0].score == 0.85


def test_search_with_budget_filter(mock_qdrant_client, sample_embedding):
    """Test search with budget constraint."""
    db = VectorDB(host="localhost", port=6333)
    db.search(sample_embedding, top_k=5, budget=1000.0)
    
    # Verify filter was applied in search call
    call_args = mock_qdrant_client.search.call_args
    assert call_args is not None
    
    # Check if query_filter exists in kwargs
    if 'query_filter' in call_args.kwargs:
        assert call_args.kwargs['query_filter'] is not None


def test_search_with_category_filter(mock_qdrant_client, sample_embedding):
    """Test search with category filter."""
    db = VectorDB(host="localhost", port=6333)
    db.search(sample_embedding, top_k=5, category="Electronics")
    
    call_args = mock_qdrant_client.search.call_args
    assert call_args is not None


def test_search_with_multiple_filters(mock_qdrant_client, sample_embedding):
    """Test search with both budget and category filters."""
    db = VectorDB(host="localhost", port=6333)
    db.search(sample_embedding, top_k=5, budget=1000.0, category="Electronics")
    
    call_args = mock_qdrant_client.search.call_args
    assert call_args is not None


def test_health_check(mock_qdrant_client):
    """Test health check functionality."""
    mock_qdrant_client.get_collections.return_value = Mock()
    
    db = VectorDB(host="localhost", port=6333)
    is_healthy = db.health_check()
    
    assert is_healthy is True


def test_health_check_failure(mock_qdrant_client):
    """Test health check when Qdrant is down."""
    mock_qdrant_client.get_collections.side_effect = Exception("Connection failed")
    
    db = VectorDB(host="localhost", port=6333)
    is_healthy = db.health_check()
    
    assert is_healthy is False


def test_get_stats(mock_qdrant_client):
    """Test getting collection statistics."""
    mock_info = Mock()
    mock_info.points_count = 100
    mock_info.vectors_count = 100
    mock_qdrant_client.get_collection.return_value = mock_info
    
    db = VectorDB(host="localhost", port=6333)
    stats = db.get_stats()
    
    assert stats['total_products'] == 100
    assert stats['collection_name'] == 'products'


def test_fin_search_engine_initialization():
    """Test FinSearchEngine initialization."""
    with patch('src.retrieval.search_engine.EmbeddingService'), \
         patch('src.retrieval.search_engine.VectorDB'):
        
        engine = FinSearchEngine()
        assert engine.embedder is not None
        assert engine.vector_db is not None


def test_fin_search_engine_search():
    """Test FinSearchEngine end-to-end search."""
    with patch('src.retrieval.search_engine.EmbeddingService') as mock_embedder_class, \
         patch('src.retrieval.search_engine.VectorDB') as mock_db_class:
        
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.embed.return_value = [0.1] * 384
        mock_embedder_class.return_value = mock_embedder
        
        mock_result = Mock()
        mock_result.score = 0.85
        mock_result.payload = {'title': 'Test', 'price': 100.0}
        
        mock_db = Mock()
        mock_db.search.return_value = [mock_result]
        mock_db_class.return_value = mock_db
        
        # Test search
        engine = FinSearchEngine()
        results = engine.search("test query", top_k=5, budget=1000.0)
        
        assert len(results) == 1
        assert results[0].score == 0.85
        assert mock_embedder.embed.called
        assert mock_db.search.called


def test_vector_db_empty_products(mock_qdrant_client):
    """Test indexing empty product list."""
    db = VectorDB(host="localhost", port=6333)
    db.index_products([], [])
    
    # Should not call upsert for empty list
    assert not mock_qdrant_client.upsert.called


def test_search_returns_empty_on_error(mock_qdrant_client, sample_embedding):
    """Test search returns empty list on error."""
    mock_qdrant_client.search.side_effect = Exception("Search failed")
    
    db = VectorDB(host="localhost", port=6333)
    results = db.search(sample_embedding, top_k=5)
    
    assert results == []
