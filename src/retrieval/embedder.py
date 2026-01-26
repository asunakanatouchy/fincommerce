"""Embedding service for text vectorization."""
import logging
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using SentenceTransformers.
    
    This service provides semantic embeddings for product descriptions and user queries.
    It includes caching, batch processing, and error handling.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model
            cache_dir: Directory to cache the model (optional)
            
        Raises:
            RuntimeError: If model fails to load
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
            self.vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"✓ Model loaded. Vector size: {self.vector_size}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not initialize embedding model: {e}")
    
    def embed(self, text: Union[str, List[str]], 
              normalize: bool = True) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text input.
        
        Args:
            text: Single text string or list of texts
            normalize: Whether to L2-normalize the embeddings
            
        Returns:
            List of floats (single text) or list of lists (multiple texts)
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text:
            raise ValueError("Text input cannot be empty")
        
        try:
            is_single = isinstance(text, str)
            texts = [text] if is_single else text
            
            # Filter out empty strings
            valid_texts = [t.strip() for t in texts if t and t.strip()]
            if not valid_texts:
                raise ValueError("No valid text after filtering empty strings")
            
            # Generate embeddings
            embeddings = self.model.encode(
                valid_texts,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            # Convert to list format
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            return embeddings[0] if is_single else embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    @lru_cache(maxsize=1000)
    def embed_cached(self, text: str) -> List[float]:
        """Cached embedding for frequently used queries.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embed(text)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a batch of texts with batching.
        
        Args:
            texts: List of text strings
            batch_size: Size of processing batches
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self.embed(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.warning(f"Failed to process batch {i//batch_size}: {e}")
                # Return zero vectors for failed batch
                all_embeddings.extend([[0.0] * self.vector_size] * len(batch))
        
        return all_embeddings
    
    def get_vector_size(self) -> int:
        """Get the dimensionality of embeddings.
        
        Returns:
            Vector dimension
        """
        return self.vector_size


# Global instance (for backward compatibility with student code)
_global_service: Optional[EmbeddingService] = None


def get_embedding(text: str) -> Optional[List[float]]:
    """Legacy function for backward compatibility.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector or None if text is empty
    """
    global _global_service
    
    if not text:
        return None
    
    if _global_service is None:
        _global_service = EmbeddingService()
    
    try:
        return _global_service.embed(text)
    except Exception as e:
        logger.error(f"get_embedding failed: {e}")
        return None


__all__ = ['EmbeddingService', 'get_embedding']