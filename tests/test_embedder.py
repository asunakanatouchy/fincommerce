"""Tests for embedding service."""
import pytest
import numpy as np
from src.retrieval.embedder import EmbeddingService, get_embedding


def test_embedding_service_initialization():
    """Test embedding service initializes correctly."""
    service = EmbeddingService()
    assert service.model is not None
    assert service.dimension == 384


def test_embed_single_text():
    """Test embedding a single text."""
    service = EmbeddingService()
    text = "laptop for software development"
    embedding = service.embed(text)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)


def test_embed_batch():
    """Test batch embedding."""
    service = EmbeddingService()
    texts = [
        "laptop for development",
        "wireless mouse",
        "mechanical keyboard"
    ]
    embeddings = service.embed_batch(texts)
    
    assert len(embeddings) == 3
    assert all(len(emb) == 384 for emb in embeddings)


def test_embed_empty_text():
    """Test embedding empty text."""
    service = EmbeddingService()
    embedding = service.embed("")
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384


def test_embed_cached():
    """Test caching functionality."""
    service = EmbeddingService()
    text = "test caching for embeddings"
    
    # First call - should compute
    emb1 = service.embed_cached(text)
    
    # Second call - should use cache
    emb2 = service.embed_cached(text)
    
    # Should return same embedding
    assert np.allclose(emb1, emb2)


def test_backward_compatible_function():
    """Test backward compatible get_embedding function."""
    text = "test backward compatibility"
    embedding = get_embedding(text)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384


def test_embedding_consistency():
    """Test that same text produces same embedding."""
    service = EmbeddingService()
    text = "consistent embedding test"
    
    emb1 = service.embed(text)
    emb2 = service.embed(text)
    
    assert np.allclose(emb1, emb2, rtol=1e-5)


def test_embedding_vector_magnitude():
    """Test that embeddings are normalized."""
    service = EmbeddingService()
    text = "vector magnitude test"
    embedding = service.embed(text)
    
    magnitude = np.linalg.norm(embedding)
    assert np.isclose(magnitude, 1.0, rtol=1e-5)
