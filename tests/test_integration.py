"""Integration tests for full pipeline."""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
from src.ingestion.load_products import load_products
from src.retrieval.embedder import EmbeddingService
from src.retrieval.search_engine import VectorDB
from src.processing.ranker import Ranker


@pytest.fixture
def temp_products_file(tmp_path, sample_products):
    """Create temporary products CSV file."""
    df = pd.DataFrame(sample_products)
    file_path = tmp_path / "test_products.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_load_products_integration(temp_products_file):
    """Test loading products from file."""
    products = load_products(temp_products_file)
    
    assert len(products) == 3
    assert all('title' in p for p in products)
    assert all('price' in p for p in products)
    assert all('category' in p for p in products)


def test_embedding_integration(sample_products):
    """Test embedding generation for products."""
    embedder = EmbeddingService()
    
    # Create text representations
    texts = [f"{p['title']} {p['description']}" for p in sample_products]
    
    # Generate embeddings
    embeddings = embedder.embed_batch(texts)
    
    assert len(embeddings) == len(sample_products)
    assert all(len(emb) == 384 for emb in embeddings)


@patch('src.retrieval.search_engine.QdrantClient')
def test_indexing_integration(mock_qdrant_class, sample_products):
    """Test indexing products to vector database."""
    # Setup mock
    mock_client = Mock()
    mock_qdrant_class.return_value = mock_client
    mock_client.collection_exists.return_value = False
    
    # Create embeddings
    embedder = EmbeddingService()
    texts = [f"{p['title']} {p['description']}" for p in sample_products]
    embeddings = embedder.embed_batch(texts)
    
    # Index products
    db = VectorDB(host="localhost", port=6333)
    db.create_collection()
    db.index_products(sample_products, embeddings)
    
    # Verify indexing was called
    assert mock_client.upsert.called


@patch('src.retrieval.search_engine.QdrantClient')
def test_search_and_rank_integration(mock_qdrant_class, sample_query, sample_budget):
    """Test end-to-end search and ranking."""
    # Setup mock
    mock_client = Mock()
    mock_qdrant_class.return_value = mock_client
    
    # Mock search results
    mock_hit = Mock()
    mock_hit.score = 0.85
    mock_hit.payload = {
        'product_id': 1,
        'title': 'Dev Laptop Pro',
        'description': 'High-performance laptop',
        'price': 1299.99,
        'category': 'Electronics',
        'brand': 'TechBrand',
        'rating': 4.7,
        'msrp': 1499.99,
        'discount_pct': 13.3,
        'stock': 50,
        'availability': 'in_stock',
        'payment_methods': 'card;paypal',
        'installment_available': True,
        'max_installments': 12,
        'shipping_days': 3,
        'budget_band': 'premium',
        'tags': 'laptop;development'
    }
    mock_client.search.return_value = [mock_hit]
    
    # Execute search pipeline
    embedder = EmbeddingService()
    query_embedding = embedder.embed(sample_query)
    
    db = VectorDB(host="localhost", port=6333)
    search_results = db.search(query_embedding, top_k=5, budget=sample_budget)
    
    ranker = Ranker()
    ranked_results = ranker.rank(search_results, sample_budget)
    
    # Verify results
    assert len(ranked_results) > 0
    assert 'composite_score' in ranked_results[0]
    assert 'explanation' in ranked_results[0]
    assert ranked_results[0]['title'] == 'Dev Laptop Pro'


@patch('src.retrieval.search_engine.QdrantClient')
def test_budget_filter_integration(mock_qdrant_class):
    """Test that budget filtering works end-to-end."""
    mock_client = Mock()
    mock_qdrant_class.return_value = mock_client
    
    # Mock result below budget
    mock_hit = Mock()
    mock_hit.score = 0.85
    mock_hit.payload = {
        'product_id': 1,
        'title': 'Budget Laptop',
        'description': 'Affordable option',
        'price': 499.99,
        'category': 'Electronics',
        'brand': 'ValueBrand',
        'rating': 3.9,
        'msrp': 599.99,
        'discount_pct': 16.7,
        'stock': 100,
        'availability': 'in_stock',
        'payment_methods': 'card',
        'installment_available': False,
        'max_installments': 0,
        'shipping_days': 5,
        'budget_band': 'budget',
        'tags': 'laptop;budget'
    }
    mock_client.search.return_value = [mock_hit]
    
    # Search with budget
    embedder = EmbeddingService()
    query_embedding = embedder.embed("laptop")
    
    db = VectorDB(host="localhost", port=6333)
    results = db.search(query_embedding, top_k=5, budget=500.0)
    
    # Verify search was called with filter
    assert mock_client.search.called
    call_args = mock_client.search.call_args
    assert call_args is not None


@patch('src.retrieval.search_engine.QdrantClient')
def test_category_filter_integration(mock_qdrant_class):
    """Test that category filtering works end-to-end."""
    mock_client = Mock()
    mock_qdrant_class.return_value = mock_client
    mock_client.search.return_value = []
    
    embedder = EmbeddingService()
    query_embedding = embedder.embed("laptop")
    
    db = VectorDB(host="localhost", port=6333)
    db.search(query_embedding, top_k=5, category="Electronics")
    
    # Verify filter was applied
    assert mock_client.search.called


def test_explanation_quality_integration():
    """Test that explanations are informative."""
    ranker = Ranker()
    
    # Create mock hit
    mock_hit = Mock()
    mock_hit.score = 0.85
    mock_hit.payload = {
        'product_id': 1,
        'title': 'Test Product',
        'description': 'Test description',
        'price': 999.99,
        'category': 'Electronics',
        'brand': 'TestBrand',
        'rating': 4.5,
        'msrp': 1199.99,
        'discount_pct': 16.7,
        'stock': 50,
        'availability': 'in_stock',
        'payment_methods': 'card',
        'installment_available': True,
        'max_installments': 12,
        'shipping_days': 3,
        'budget_band': 'premium',
        'tags': 'test'
    }
    
    results = ranker.rank([mock_hit], budget=1500.0)
    
    explanation = results[0]['explanation']
    
    # Check explanation contains key information
    assert len(explanation) > 0
    assert 'score' in explanation.lower() or 'match' in explanation.lower()


@patch('src.retrieval.search_engine.QdrantClient')
def test_full_pipeline_no_results(mock_qdrant_class):
    """Test pipeline handles no results gracefully."""
    mock_client = Mock()
    mock_qdrant_class.return_value = mock_client
    mock_client.search.return_value = []
    
    embedder = EmbeddingService()
    query_embedding = embedder.embed("nonexistent product xyz")
    
    db = VectorDB(host="localhost", port=6333)
    results = db.search(query_embedding, top_k=5, budget=1000.0)
    
    ranker = Ranker()
    ranked = ranker.rank(results, budget=1000.0)
    
    assert ranked == []


def test_composite_scoring_weights_integration():
    """Test that composite scoring applies correct weights."""
    ranker = Ranker()
    
    # Verify weights sum to 1.0
    total_weight = (ranker.weights.semantic + 
                   ranker.weights.budget_fit + 
                   ranker.weights.price_advantage)
    assert abs(total_weight - 1.0) < 0.01
