"""Qdrant vector database integration for semantic search."""
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, Range, SearchRequest
)
from qdrant_client.http.exceptions import UnexpectedResponse

from .embedder import EmbeddingService

logger = logging.getLogger(__name__)


class VectorDB:
    """Qdrant vector database wrapper with connection management and error handling."""
    
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 collection_name: str = "products", vector_size: int = 384,
                 api_key: Optional[str] = None, timeout: int = 30):
        """Initialize Qdrant client with connection management.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of collection to use
            vector_size: Dimension of embeddings
            api_key: Optional API key for authentication
            timeout: Connection timeout in seconds
            
        Raises:
            ConnectionError: If cannot connect to Qdrant
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout = timeout
        
        try:
            self.client = QdrantClient(
                host=host, 
                port=port,
                api_key=api_key,
                timeout=timeout
            )
            # Test connection
            self.client.get_collections()
            logger.info(f"✓ Connected to Qdrant at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Cannot connect to Qdrant at {host}:{port}: {e}")
    
    def create_collection(self, recreate: bool = False) -> bool:
        """Create collection if it doesn't exist.
        
        Args:
            recreate: If True, delete and recreate collection
            
        Returns:
            True if successful
        """
        try:
            if recreate:
                try:
                    self.client.delete_collection(self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                except:
                    pass
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if exists and not recreate:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✓ Created collection '{self.collection_name}' with {self.vector_size}D vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def index_products(self, products: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """Index products to Qdrant with batching.
        
        Args:
            products: List of products with 'embedding' field
            batch_size: Number of products per batch
            
        Returns:
            Number of products successfully indexed
        """
        if not products:
            logger.warning("No products to index")
            return 0
        
        indexed_count = 0
        
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            points = []
            
            for product in batch:
                if 'embedding' not in product:
                    logger.warning(f"Product {product.get('id', 'unknown')} missing embedding")
                    continue
                
                try:
                    point = PointStruct(
                        id=product.get('id', indexed_count),
                        vector=product['embedding'],
                        payload={
                            'product_id': product.get('id'),
                            'title': product.get('title', ''),
                            'description': product.get('description', ''),
                            'price': float(product.get('price', 0)),
                            'category': product.get('category', ''),
                            'brand': product.get('brand', ''),
                            'rating': float(product.get('rating', 0)),
                            'msrp': float(product.get('msrp', 0)),
                            'discount_pct': float(product.get('discount_pct', 0)),
                            'stock': int(product.get('stock', 0)),
                            'availability': product.get('availability', ''),
                            'payment_methods': product.get('payment_methods', ''),
                            'installment_available': product.get('installment_available', False),
                            'max_installments': int(product.get('max_installments', 0)),
                            'shipping_days': int(product.get('shipping_days', 0)),
                            'budget_band': product.get('budget_band', ''),
                            'tags': product.get('tags', ''),
                        }
                    )
                    points.append(point)
                except Exception as e:
                    logger.warning(f"Failed to create point for product {product.get('id')}: {e}")
                    continue
            
            if points:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    indexed_count += len(points)
                    logger.info(f"Indexed batch {i//batch_size + 1}: {len(points)} products")
                except Exception as e:
                    logger.error(f"Failed to index batch {i//batch_size + 1}: {e}")
        
        logger.info(f"✓ Total indexed: {indexed_count}/{len(products)} products")
        return indexed_count
    
    def search(self, query_vector: List[float], max_budget: Optional[float] = None,
               category: Optional[str] = None, top_k: int = 5,
               score_threshold: Optional[float] = None) -> List[Any]:
        """Perform constraint-aware semantic search.
        
        Args:
            query_vector: Query embedding vector
            max_budget: Maximum price filter (optional)
            category: Category filter (optional)
            top_k: Number of results to return
            score_threshold: Minimum similarity score (optional)
            
        Returns:
            List of search results with scores and payloads
            
        Raises:
            ValueError: If query_vector is invalid
        """
        if not query_vector or len(query_vector) != self.vector_size:
            raise ValueError(f"Invalid query vector. Expected {self.vector_size}D")
        
        # Build filter conditions
        filter_conditions = []
        
        if max_budget is not None:
            filter_conditions.append(
                FieldCondition(
                    key="price",
                    range=Range(lte=max_budget)
                )
            )
        
        if category:
            filter_conditions.append(
                FieldCondition(
                    key="category",
                    match={"value": category}
                )
            )
        
        query_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        try:
            response = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            logger.info(f"Search returned {len(response)} results")
            return response
            
        except UnexpectedResponse as e:
            logger.error(f"Qdrant search failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'collection_name': self.collection_name,
                'points_count': info.points_count,
                'vector_size': self.vector_size,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check if Qdrant connection is healthy.
        
        Returns:
            True if healthy
        """
        try:
            self.client.get_collections()
            return True
        except:
            return False


class FinSearchEngine:
    """High-level search engine combining embedding and vector search.
    
    This is a convenience wrapper that maintains backward compatibility
    with the student's original API.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize search engine with embedding service and vector DB.
        
        Args:
            host: Qdrant host
            port: Qdrant port
        """
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDB(host=host, port=port)
        self.collection_name = self.vector_db.collection_name
    
    def search(self, query_text: str, max_budget: float, 
               top_k: int = 5, category: Optional[str] = None) -> List[Any]:
        """Perform end-to-end semantic search with constraints.
        
        Args:
            query_text: User query string
            max_budget: Maximum price constraint
            top_k: Number of results
            category: Optional category filter
            
        Returns:
            List of search results
        """
        # Convert query to vector
        query_vector = self.embedding_service.embed(query_text)
        
        # Search with constraints
        results = self.vector_db.search(
            query_vector=query_vector,
            max_budget=max_budget,
            category=category,
            top_k=top_k
        )
        
        return results


__all__ = ['VectorDB', 'FinSearchEngine']
        