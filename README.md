
# 🛍️ FinCommerce: Context-Aware Product Discovery & Recommendations

Smart, explainable, and financially sensitive product search for e-commerce. Built for Hackathon Use Case 2.


---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Qdrant Vector DB (Recommended)
```bash
docker-compose up qdrant -d
```

### 3. Run API Server
```bash
uvicorn api.main:app --reload
```

### 4. Run Streamlit UI (Optional)
```bash
streamlit run ui/streamlit_app.py
```

### 5. Access API Docs
- [http://localhost:8000/docs](http://localhost:8000/docs)

### 6. Example Search Request
```json
{
  "query": "laptop for development",
  "budget": 1500.0,
  "top_k": 5,
  "category": "Electronics"
}
```

### 7. Feedback Endpoint (Learning Loop)
```json
POST /feedback
{
  "user_id": "user123",
  "action": "click",
  "product_id": "1",
  "query": "laptop for development",
  "budget": 1500.0,
  "timestamp": 1700000000.0,
  "extra": {}
}
```

### 8. Alternatives Suggestion
- If no products fit the budget, the API returns an `alternatives` field with close matches (e.g., slightly over budget).

### 9. Explainable Recommendations
- Each product result includes an `explanation` field describing why it was recommended.

---

## 🏗️ Architecture Overview

```
fincommerce/
├── api/                # FastAPI app & endpoints
│   ├── main.py         # API logic
│   └── schemas/        # Pydantic models
├── config/             # YAML & env config
├── src/
│   ├── core/           # Config loader
│   ├── retrieval/      # Embedding & vector search
│   ├── processing/     # Ranking & explanations
│   ├── ingestion/      # Data loading
│   └── utils/          # Utilities
├── ui/                 # Streamlit UI
├── tests/              # Pytest suite
└── data/               # Product catalog CSV
```

**Key Flow:**
1. User submits search (query, budget, etc.)
2. Query embedded (SentenceTransformers)
3. Vector search (Qdrant)
4. Results filtered/ranked by financial context
5. Explanations generated for each result
6. Alternatives suggested if needed
7. Feedback loop collects user actions

---

## 🧠 Features & Capabilities

- **Discovery for vague intents**: Semantic search for queries like "laptop for dev", "gift", "cheap but reliable".
- **Constraint-aware recommendations**: Budget, payment, and delivery constraints respected.
- **Explainable ranking**: Every result includes a clear, actionable explanation.
- **Alternatives generation**: If nothing fits, close substitutes are suggested and explained.
- **Feedback loop**: User actions (clicks, add-to-cart, etc.) collected for future learning.
- **Success indicators**: Engagement metrics tracked in UI (clicks, viewed products).
- **Sensitive data**: Only minimal, anonymized feedback stored (in-memory for demo).
- **Diversity**: No over-personalization; results remain broad and exploratory.

---

## 📊 Ranking Algorithm

**Composite Score Formula:**
```
Score = (0.6 × semantic) + (0.3 × budget_fit) + (0.1 × price_advantage)
```
Where:
- **semantic**: Cosine similarity from vector search (0-1)
- **budget_fit**: 1.0 if within budget, 0.5 if over
- **price_advantage**: (budget - price) / budget (savings ratio)

---

## 🧩 Data & Constraints

**Product CSV Columns:**
- Required: `id`, `title`, `description`, `price`, `category`, `brand`, `rating`
- Financial: `msrp`, `discount_pct`, `stock`, `availability`, `payment_methods`, `installment_available`, `max_installments`, `shipping_days`, `budget_band`, `tags`

---

## 🧪 Testing

Run all tests (including feedback & alternatives):
```bash
pytest --cov=src --cov-report=html
```

---

## 🖥️ UI (Streamlit)

- Run: `streamlit run ui/streamlit_app.py`
- Features: Search, budget slider, category filter, alternatives, engagement metrics

---

## 📄 License

MIT License

---

**Built with ❤️ for smarter e-commerce search**

