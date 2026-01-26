"""Streamlit UI for FinCommerce Vector Search."""
import streamlit as st
import requests
import json
import time
from typing import Dict, List, Optional

# ═════════════════════════════════════════════════════════════
# Page Configuration
# ═════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FinCommerce Search",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = st.secrets.get("API_URL", "http://localhost:8000") if hasattr(st, "secrets") else "http://localhost:8000"

# ═════════════════════════════════════════════════════════════
# Custom CSS
# ═════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .product-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        background: #f9f9f9;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .score-high { background: #4caf50; color: white; }
    .score-medium { background: #ff9800; color: white; }
    .score-low { background: #f44336; color: white; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# Session State Initialization
# ═════════════════════════════════════════════════════════════
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'clicked_products' not in st.session_state:
    st.session_state.clicked_products = set()

# ═════════════════════════════════════════════════════════════
# Helper Functions
# ═════════════════════════════════════════════════════════════
def check_api_health() -> bool:
    """Check if API is reachable."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def track_click(product_id: str):
    """Track product click for feedback loop."""
    st.session_state.clicked_products.add(product_id)

def get_score_color(score: float) -> str:
    """Get color class for score badge."""
    if score >= 0.7:
        return "score-high"
    elif score >= 0.5:
        return "score-medium"
    else:
        return "score-low"

def search_products(query: str, budget: float, top_k: int = 10, 
                   category: Optional[str] = None) -> Dict:
    """Call search API."""
    request_data = {
        "query": query,
        "budget": budget,
        "top_k": top_k
    }
    if category:
        request_data["category"] = category
    
    response = requests.post(
        f"{API_URL}/search",
        json=request_data,
        timeout=30
    )
    response.raise_for_status()
    return response.json()

def get_alternatives(query: str, budget: float, original_results: int) -> Dict:
    """Get alternative suggestions if few results found."""
    if original_results < 3:
        # Increase budget by 30% for alternatives
        alt_budget = budget * 1.3
        return search_products(query, alt_budget, top_k=5)
    return None

# ═════════════════════════════════════════════════════════════
# Header
# ═════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">🛍️ FinCommerce - Smart Product Search</div>', unsafe_allow_html=True)
st.markdown("**Semantic search with budget awareness** • Find what you mean, not just keywords")

# API Health Check
api_healthy = check_api_health()
if api_healthy:
    st.success("✅ System online")
else:
    st.error("❌ Cannot reach API. Make sure it's running: `python api/main.py`")
    st.stop()

st.divider()

# ═════════════════════════════════════════════════════════════
# Sidebar - Search Configuration
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🔍 Search Settings")
    
    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g., laptop for development, gift under budget, reliable but cheap...",
        help="Use natural language - the system understands intent"
    )
    
    budget = st.number_input(
        "Maximum Budget (€)",
        min_value=0.0,
        max_value=10000.0,
        value=1500.0,
        step=50.0,
        help="Products over this price will be filtered out"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Results", 3, 20, 10)
    with col2:
        show_alternatives = st.checkbox("Show alternatives", value=True,
                                       help="Show higher-priced alternatives if few results found")
    
    category = st.selectbox(
        "Category (optional)",
        ["All Categories", "Electronics", "Fashion", "Books", "Beauty", 
         "Sports", "Home & Kitchen"],
        help="Filter by category"
    )
    category = None if category == "All Categories" else category
    
    search_button = st.button("🚀 Search", use_container_width=True, type="primary")
    
    st.divider()
    
    # Search Tips
    with st.expander("💡 Search Tips"):
        st.markdown("""
        **Try these searches:**
        - "laptop for development"
        - "cheap but reliable"
        - "gift for runner"
        - "home office setup"
        - "beginner friendly"
        
        **The system understands:**
        - Synonyms (cheap = affordable = budget)
        - Context (development = coding = programming)
        - Constraints (respects your budget automatically)
        """)
    
    # Scoring Formula
    with st.expander("📊 How Ranking Works"):
        st.markdown("""
        **Composite Score Formula:**
        ```
        Score = (0.6 × Semantic) + (0.3 × Budget Fit) + (0.1 × Price Advantage)
        ```
        
        - **Semantic:** How well it matches your query
        - **Budget Fit:** Is it within your budget?
        - **Price Advantage:** How much you save
        """)

# ═════════════════════════════════════════════════════════════
# Main Content
# ═════════════════════════════════════════════════════════════

if search_button and query:
    # Validate input
    if not query.strip():
        st.warning("⚠️ Please enter a search query")
        st.stop()
    
    # Add to history
    st.session_state.search_history.append({
        "query": query,
        "budget": budget,
        "timestamp": time.time()
    })
    
    # Search
    with st.spinner("🔎 Searching products..."):
        try:
            start_time = time.time()
            results = search_products(query, budget, top_k, category)
            search_time = time.time() - start_time
            
            # Display Results Header
            st.subheader(f"Found {results['total_results']} products")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Query", f'"{query}"')
            with col2:
                st.metric("Budget", f"€{budget:.2f}")
            with col3:
                st.metric("Search Time", f"{results.get('execution_time_ms', 0):.0f}ms")
            
            st.divider()
            
            # Display Products
            if results["results"]:
                for idx, product in enumerate(results["results"], 1):
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"### #{idx} {product['title']}")
                            st.markdown(f"*{product['description'][:150]}...*")
                            st.caption(f"**Category:** {product['category']} | **Brand:** {product.get('brand', 'N/A')}")
                        
                        with col2:
                            st.metric("Price", f"€{product['price']:.2f}")
                            if product.get('rating'):
                                st.metric("Rating", f"{product['rating']:.1f}⭐")
                        
                        with col3:
                            score = product.get('composite_score', 0)
                            score_class = get_score_color(score)
                            similarity = int(product['semantic_score'] * 100)
                            
                            st.metric("Match", f"{similarity}%")
                            st.markdown(f'<span class="score-badge {score_class}">Score: {score:.2f}</span>', 
                                      unsafe_allow_html=True)
                        
                        # Explanation
                        st.info(f"💡 **Why this?** {product['explanation']}")
                        
                        # Additional Details
                        with st.expander("📦 More Details"):
                            details_col1, details_col2 = st.columns(2)
                            with details_col1:
                                if product.get('installment_available'):
                                    st.success(f"✓ Installments available (up to {product.get('max_installments', 0)} months)")
                                if product.get('msrp'):
                                    discount = product.get('discount_pct', 0)
                                    if discount > 0:
                                        st.success(f"✓ {discount:.1f}% off (was €{product.get('msrp'):.2f})")
                            with details_col2:
                                if product.get('shipping_days'):
                                    st.info(f"🚚 Ships in {product['shipping_days']} days")
                                if product.get('budget_band'):
                                    st.info(f"💰 {product['budget_band'].title()} tier")
                        
                        # Track interaction
                        if st.button(f"View Details #{idx}", key=f"view_{product['product_id']}"):
                            track_click(product['product_id'])
                            st.toast(f"✓ Clicked: {product['title']}")
                        
                        st.divider()
                
                # Alternatives (E.4 - Alternatives generation)
                if show_alternatives and results['total_results'] < 3:
                    st.warning("⚠️ Few results found within budget. Showing alternatives...")
                    
                    with st.spinner("Finding alternatives..."):
                        try:
                            alt_results = get_alternatives(query, budget, results['total_results'])
                            if alt_results and alt_results['results']:
                                st.subheader("🔄 Alternative Suggestions (Slightly Over Budget)")
                                
                                for idx, product in enumerate(alt_results['results'][:3], 1):
                                    over_budget = product['price'] - budget
                                    st.markdown(f"**{idx}. {product['title']}** - €{product['price']:.2f} (€{over_budget:.2f} over)")
                                    st.caption(product['explanation'])
                                    st.divider()
                        except Exception as e:
                            st.error(f"Could not fetch alternatives: {e}")
                
            else:
                st.warning("😕 No products found matching your criteria")
                st.markdown("""
                **Try:**
                - Increasing your budget
                - Using different keywords
                - Removing category filter
                - Simplifying your query
                """)
        
        except requests.exceptions.Timeout:
            st.error("⏱️ Search took too long. Try reducing results or check server performance.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to API. Make sure FastAPI is running on port 8000.")
        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")

elif not query and search_button:
    st.info("👆 Enter a search query in the sidebar to get started")

# ═════════════════════════════════════════════════════════════
# Footer - System Stats & History
# ═════════════════════════════════════════════════════════════
st.divider()

col1, col2 = st.columns(2)

with col1:
    with st.expander("📊 System Statistics"):
        try:
            stats = requests.get(f"{API_URL}/stats", timeout=2).json()
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("Products Indexed", stats.get("points_count", 0))
            with stat_col2:
                st.metric("Vector Dimension", stats.get("vector_size", 0))
            with stat_col3:
                st.metric("Model", stats.get("embedding_model", "N/A"))
        except:
            st.warning("Could not fetch system stats")

with col2:
    with st.expander("🕒 Search History"):
        if st.session_state.search_history:
            recent = st.session_state.search_history[-5:]
            for item in reversed(recent):
                st.caption(f"• \"{item['query']}\" (€{item['budget']:.0f})")
            
            if st.button("Clear History"):
                st.session_state.search_history = []
                st.rerun()
        else:
            st.caption("No searches yet")

# Metrics for Success Indicators (F)
if st.session_state.clicked_products:
    st.caption(f"📈 Engagement: {len(st.session_state.clicked_products)} products viewed this session")

st.divider()
st.caption("🚀 FinCommerce Vector Search • Built with Qdrant + SentenceTransformers + FastAPI • Hackathon Project")
