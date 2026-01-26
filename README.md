# FinCommerce - Semantic Product Search with Budget Awareness

Context-aware e-commerce search engine powered by Qdrant vector database and semantic embeddings.

## 🚀 Features

- **Semantic Search**: Natural language product discovery using sentence transformers
- **Budget-Aware Filtering**: Constraint-based recommendations that respect user budgets
- **Composite Ranking**: Multi-factor scoring (semantic similarity + budget fit + price advantage)
- **Explainable Results**: Clear explanations for why each product is recommended
- **Production-Ready**: FastAPI backend with proper logging, error handling, and Docker support

## 📋 Use Case

**Context-Aware FinCommerce Engine** - Smart product discovery with financial constraints.

Instead of keyword matching, the system understands:
- "laptop for development" → finds dev-friendly laptops
- "cheap but reliable" → balances price and quality
- "gift under €500" → respects budget constraints

## 🏗️ Architecture

```
fincomerce/
├── api/                    # FastAPI application
│   ├── main.py            # API endpoints
│   └── schemas/           # Pydantic models
├── config/                # Configuration files
│   ├── settings.yaml      # Main configuration
│   └── .env.example       # Environment variables
├── src/
│   ├── core/              # Core configuration
│   ├── retrieval/         # Embedding & vector search
│   │   ├── embedder.py    # SentenceTransformers service
│   │   └── search_engine.py  # Qdrant integration
│   ├── processing/        # Ranking & explanations
│   │   └── ranker.py      # Composite scoring
│   ├── ingestion/         # Data loading
│   └── utils/             # Utilities
├── ui/                    # Streamlit interface
├── tests/                 # Unit tests
└── data/                  # Product catalog
```

## 🔧 Installation

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (recommended)

### Quick Start with Docker

```bash
# Start all services (Qdrant + API + UI)
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Access UI
open http://localhost:8501
```

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker-compose up qdrant -d

# Configure environment
cp config/.env.example config/.env

# Run API
python api/main.py

# Run UI (separate terminal)
streamlit run ui/streamlit_app.py
```

## 📊 Product Data

The system expects CSV with these columns:

**Required:**
- `id`, `title`, `description`, `price`, `category`, `brand`, `rating`

**Financial Metadata (for constraints):**
- `msrp`, `discount_pct`, `stock`, `availability`
- `payment_methods`, `installment_available`, `max_installments`
- `shipping_days`, `budget_band`, `tags`

## 🔍 API Usage

### Search Products

```bash
POST /search
Content-Type: application/json

{
  "query": "laptop for development",
  "budget": 1500.0,
  "top_k": 5,
  "category": "Electronics"
}
```

**Response:**
```json
{
  "query": "laptop for development",
  "budget": 1500.0,
  "total_results": 5,
  "results": [
    {
      "title": "Dev Laptop 14",
      "price": 1199.0,
      "semantic_score": 0.8542,
      "composite_score": 0.7913,
      "explanation": "Matches your intent (85.4%) and is €301 under budget."
    }
  ],
  "execution_time_ms": 45.2
}
```

## 📈 Ranking Algorithm

**Composite Score Formula:**
```
Score = (0.6 × semantic) + (0.3 × budget_fit) + (0.1 × price_advantage)
```

Where:
- **semantic**: Cosine similarity from vector search (0-1)
- **budget_fit**: 1.0 if within budget, 0.5 if over
- **price_advantage**: (budget - price) / budget (savings ratio)

## 🧩 Chunking Strategy

**Current:** Disabled (products have short descriptions ~100 chars)

**When to Enable:**
- Product descriptions > 512 tokens
- Integration of user reviews (aggregated long-form text)
- Multi-language catalogs requiring cross-lingual embeddings

**Configuration** (`config/settings.yaml`):
```yaml
chunking:
  enabled: true
  chunk_size: 256
  chunk_overlap: 50
  strategy: "sentence"
```

**Implementation Approaches:**

1. **Sentence-Based Chunking** (Recommended for product descriptions)
2. **Fixed-Size Chunking** (For uniform review text)
3. **Semantic Chunking** (For long reviews or multi-attribute products)

## 🧪 Testing

```bash
pytest --cov=src --cov-report=html
```

## 📄 License

MIT License

---

**Built with ❤️ for smarter e-commerce search**
