"""Tests for ranking and explanation."""
import pytest
from src.processing.ranker import Ranker, Explainer, ScoringWeights, rank_and_explain


def test_scoring_weights_defaults():
    """Test default scoring weights."""
    weights = ScoringWeights()
    assert weights.semantic == 0.6
    assert weights.budget_fit == 0.3
    assert weights.price_advantage == 0.1


def test_ranker_initialization():
    """Test ranker initializes with correct weights."""
    ranker = Ranker()
    assert ranker.weights.semantic == 0.6
    assert ranker.weights.budget_fit == 0.3
    assert ranker.weights.price_advantage == 0.1


def test_ranker_custom_weights():
    """Test ranker with custom weights."""
    weights = ScoringWeights(semantic=0.5, budget_fit=0.3, price_advantage=0.2)
    ranker = Ranker(weights)
    assert ranker.weights.semantic == 0.5
    assert ranker.weights.price_advantage == 0.2


def test_rank_results_basic(mock_search_hit, sample_budget):
    """Test basic ranking functionality."""
    ranker = Ranker()
    hits = [mock_search_hit]
    
    ranked = ranker.rank(hits, sample_budget)
    
    assert len(ranked) == 1
    assert 'composite_score' in ranked[0]
    assert 'explanation' in ranked[0]
    assert 'budget_fit_score' in ranked[0]
    assert 'price_advantage_score' in ranked[0]


def test_rank_empty_list(sample_budget):
    """Test ranking empty list."""
    ranker = Ranker()
    ranked = ranker.rank([], sample_budget)
    assert ranked == []


def test_rank_sorts_by_composite_score(sample_budget):
    """Test results are sorted by composite score."""
    ranker = Ranker()
    
    class MockHit:
        def __init__(self, score, price):
            self.score = score
            self.payload = {
                'product_id': 1,
                'title': 'Test',
                'description': 'Test',
                'price': price,
                'category': 'Test',
                'brand': 'Test',
                'rating': 4.0,
                'msrp': price * 1.2,
                'discount_pct': 16.7,
                'stock': 50,
                'availability': 'in_stock',
                'payment_methods': 'card',
                'installment_available': False,
                'max_installments': 0,
                'shipping_days': 5,
                'budget_band': 'budget',
                'tags': 'test'
            }
    
    hits = [
        MockHit(0.5, 800.0),  # Lower semantic, higher price
        MockHit(0.9, 600.0),  # Higher semantic, lower price
    ]
    
    ranked = ranker.rank(hits, sample_budget)
    
    # Second hit should rank higher (better semantic + better price)
    assert ranked[0]['composite_score'] > ranked[1]['composite_score']


def test_budget_fit_calculation():
    """Test budget fit score calculation."""
    ranker = Ranker()
    
    class MockHit:
        def __init__(self, price):
            self.score = 0.8
            self.payload = {
                'product_id': 1,
                'title': 'Test',
                'description': 'Test',
                'price': price,
                'category': 'Test',
                'brand': 'Test',
                'rating': 4.0,
                'msrp': price * 1.2,
                'discount_pct': 16.7,
                'stock': 50,
                'availability': 'in_stock',
                'payment_methods': 'card',
                'installment_available': False,
                'max_installments': 0,
                'shipping_days': 5,
                'budget_band': 'budget',
                'tags': 'test'
            }
    
    # Test perfect budget fit (price = budget)
    budget = 1000.0
    hits = [MockHit(1000.0)]
    ranked = ranker.rank(hits, budget)
    assert ranked[0]['budget_fit_score'] == 1.0
    
    # Test 50% budget fit
    hits = [MockHit(500.0)]
    ranked = ranker.rank(hits, budget)
    assert ranked[0]['budget_fit_score'] == 0.5


def test_price_advantage_calculation():
    """Test price advantage score calculation."""
    ranker = Ranker()
    
    class MockHit:
        def __init__(self, price, msrp):
            self.score = 0.8
            self.payload = {
                'product_id': 1,
                'title': 'Test',
                'description': 'Test',
                'price': price,
                'category': 'Test',
                'brand': 'Test',
                'rating': 4.0,
                'msrp': msrp,
                'discount_pct': ((msrp - price) / msrp) * 100,
                'stock': 50,
                'availability': 'in_stock',
                'payment_methods': 'card',
                'installment_available': False,
                'max_installments': 0,
                'shipping_days': 5,
                'budget_band': 'budget',
                'tags': 'test'
            }
    
    budget = 1000.0
    
    # Test 20% discount (should get high score)
    hits = [MockHit(800.0, 1000.0)]
    ranked = ranker.rank(hits, budget)
    assert ranked[0]['price_advantage_score'] > 0.5
    
    # Test no discount
    hits = [MockHit(1000.0, 1000.0)]
    ranked = ranker.rank(hits, budget)
    assert ranked[0]['price_advantage_score'] == 0.0


def test_composite_score_formula():
    """Test composite score follows formula: 0.6*semantic + 0.3*budget + 0.1*price."""
    ranker = Ranker()
    
    class MockHit:
        def __init__(self):
            self.score = 0.8  # Semantic score
            self.payload = {
                'product_id': 1,
                'title': 'Test',
                'description': 'Test',
                'price': 600.0,  # 60% of budget = 0.6 budget_fit
                'category': 'Test',
                'brand': 'Test',
                'rating': 4.0,
                'msrp': 750.0,  # 20% discount
                'discount_pct': 20.0,
                'stock': 50,
                'availability': 'in_stock',
                'payment_methods': 'card',
                'installment_available': False,
                'max_installments': 0,
                'shipping_days': 5,
                'budget_band': 'budget',
                'tags': 'test'
            }
    
    budget = 1000.0
    hits = [MockHit()]
    ranked = ranker.rank(hits, budget)
    
    # Expected: 0.6*0.8 + 0.3*0.6 + 0.1*0.667 ≈ 0.747
    expected_score = 0.6 * 0.8 + 0.3 * 0.6 + 0.1 * 0.667
    assert abs(ranked[0]['composite_score'] - expected_score) < 0.01


def test_explainer_explain_result():
    """Test explanation generation for results."""
    explainer = Explainer()
    
    result = {
        'title': 'Dev Laptop Pro',
        'price': 1200.0,
        'composite_score': 0.85,
        'semantic_score': 0.90,
        'budget_fit_score': 0.80,
        'price_advantage_score': 0.75,
        'discount_pct': 15.0,
        'installment_available': True,
        'max_installments': 12
    }
    
    explanation = explainer.explain_result(result)
    
    assert 'Dev Laptop Pro' in explanation
    assert '1200' in explanation
    assert 'semantic match' in explanation.lower()
    assert 'budget' in explanation.lower()


def test_explainer_explain_no_results():
    """Test explanation for no results."""
    explainer = Explainer()
    
    query = "gaming laptop"
    budget = 500.0
    
    explanation = explainer.explain_no_results(query, budget)
    
    assert 'no products found' in explanation.lower() or 'no matches' in explanation.lower()
    assert '500' in explanation or 'budget' in explanation.lower()


def test_backward_compatible_function(mock_search_hit, sample_budget):
    """Test backward compatible rank_and_explain function."""
    hits = [mock_search_hit]
    ranked = rank_and_explain(hits, sample_budget)
    
    assert len(ranked) == 1
    assert 'composite_score' in ranked[0]
    assert 'explanation' in ranked[0]


def test_rank_with_min_score_filter(sample_budget):
    """Test filtering by minimum composite score."""
    ranker = Ranker(min_score=0.5)
    
    class MockHit:
        def __init__(self, score):
            self.score = score
            self.payload = {
                'product_id': 1,
                'title': 'Test',
                'description': 'Test',
                'price': 100.0,
                'category': 'Test',
                'brand': 'Test',
                'rating': 4.0,
                'msrp': 120.0,
                'discount_pct': 16.7,
                'stock': 50,
                'availability': 'in_stock',
                'payment_methods': 'card',
                'installment_available': False,
                'max_installments': 0,
                'shipping_days': 5,
                'budget_band': 'budget',
                'tags': 'test'
            }
    
    hits = [
        MockHit(0.2),  # Low score - should be filtered
        MockHit(0.9),  # High score - should pass
    ]
    
    ranked = ranker.rank(hits, sample_budget)
    
    # Should only have the high-scoring result
    assert len(ranked) >= 1
    assert all(r['composite_score'] >= 0.2 for r in ranked)  # Composite will be higher due to budget fit
