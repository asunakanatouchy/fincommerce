import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

"""
Ingest and index all products from products.csv into Qdrant using the project's embedder and vector DB logic.
"""
from pathlib import Path
from src.ingestion.load_products import load_products
from src.retrieval.embedder import EmbeddingService
from src.retrieval.search_engine import VectorDB
from src.core import CONFIG
import tqdm

# Load config
embedding_config = CONFIG.get('embeddings', {})
qdrant_config = CONFIG.get('qdrant', {})

# Initialize embedding service
embedder = EmbeddingService(
    model_name=embedding_config.get('model_name', 'all-MiniLM-L6-v2')
)

# Initialize vector DB
vector_db = VectorDB(
    host=qdrant_config.get('host', 'localhost'),
    port=qdrant_config.get('port', 6333),
    collection_name=qdrant_config.get('collection_name', 'products'),
    vector_size=qdrant_config.get('vector_size', 384)
)

# Load products
df = load_products()
print(f"Loaded {len(df)} products. Embedding and indexing...")


# Prepare products for Qdrant with improved embedding and normalization
products = []
for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    # Normalize and combine fields for embedding
    title = str(row['title']).strip().lower()
    desc = str(row['description']).strip().lower()
    tags = ' '.join(row['tags']) if isinstance(row.get('tags', []), list) else str(row.get('tags', '')).replace(';', ' ')
    tags = tags.strip().lower()
    category = str(row['category']).strip().lower()
    brand = str(row.get('brand', '')).strip().lower()
    embed_input = f"{title} {desc} {tags} {category} {brand}".strip()
    embedding = embedder.embed(embed_input)
    product = row.to_dict()
    product['embedding'] = embedding
    # Also normalize fields for search/fuzzy fallback
    product['title'] = title
    product['description'] = desc
    product['tags'] = tags
    product['category'] = category
    product['brand'] = brand
    products.append(product)
    if _ % 100 == 0:
        print(f"Indexed {_+1} products so far...")

# Index in Qdrant
vector_db.create_collection(recreate=False)
vector_db.index_products(products, batch_size=100)
print(f"Indexed {len(products)} products into Qdrant!")
