from typing import Dict, Any

class Explainer:
    """Generate detailed explanations for recommendations."""

    @staticmethod
    def explain_no_results(query: str, budget: float, filters: dict) -> str:
        """Generate explanation when no results are found."""
        explanation = (
            f"No products matched your query '{query}' within your budget of €{budget:.2f}. "
            f"We suggest relaxing your filters or increasing your budget. "
            f"Filters applied: {filters}"
        )
        return explanation

    @staticmethod
    def explain_result(result: Dict[str, Any], query: str, budget: float, *args, **kwargs) -> str:
        """Generate detailed explanation for a single result.
        Args:
            result: Ranked product result
            query: Original user query
            budget: User budget
        Returns:
            Detailed explanation string
        """
        parts = [
            f"**Why we recommend {result['title']}:**",
            f"",
            f"1. **Semantic Match:** {result['semantic_score']*100:.1f}% relevance to your query '{query}'",
            f"2. **Price:** €{result['price']:.2f} (Budget: €{budget:.2f})",
        ]
        if result['price'] <= budget:
            savings = budget - result['price']
            parts.append(f"   ✓ Under budget by €{savings:.2f}")
        else:
            overage = result['price'] - budget
            parts.append(f"   ⚠ Over budget by €{overage:.2f}")
        if result.get('installment_available'):
            parts.append(f"3. **Payment:** Installments available (up to {result['max_installments']} months)")
        if result.get('discount_pct', 0) > 0:
            parts.append(f"4. **Discount:** {result['discount_pct']:.1f}% off (MSRP: €{result.get('msrp', 0):.2f})")
        if result.get('rating', 0) > 0:
            parts.append(f"5. **Rating:** {result['rating']:.1f}/5.0 stars")
        parts.append(f"")
        parts.append(f"**Composite Score:** {result['composite_score']:.2f}/1.00")
        return "\n".join(parts)
"""Ranking and explanation logic for search results."""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Weights for composite scoring formula."""
    semantic: float = 0.6
    budget_fit: float = 0.3
    price_advantage: float = 0.1
    
    def validate(self) -> bool:
        """Validate that weights sum to ~1.0."""
        total = self.semantic + self.budget_fit + self.price_advantage
        return abs(total - 1.0) < 0.01


class Ranker:
    """Rank search results using composite scoring with financial awareness."""
    
    def __init__(self, weights: Optional[ScoringWeights] = None, min_score: float = 0.0):
        """Initialize ranker with scoring weights and min_score.
        Args:
            weights: Scoring weights (defaults to 60/30/10 formula)
            min_score: Minimum composite score threshold
        """
        self.weights = weights or ScoringWeights()
        self.min_score = min_score
        if not self.weights.validate():
            logger.warning(f"Scoring weights don't sum to 1.0: {self.weights}")

    def rank(self, search_hits: List[Any], user_budget: float, min_score: Optional[float] = None, **kwargs) -> List[Dict[str, Any]]:
        if not search_hits:
            logger.info("No search hits to rank")
            return []

        ranked_results = []
        min_score_val = self.min_score if min_score is None else min_score

        for hit in search_hits:
            try:
                payload = hit.payload if hasattr(hit, 'payload') else hit
                semantic_score = hit.score if hasattr(hit, 'score') else 0.0
                price = float(payload.get("price", 0))
                budget_fit = 1.0 if price <= user_budget else 0.5
                price_advantage = max(0, (user_budget - price) / user_budget) if user_budget > 0 else 0
                final_score = (
                    self.weights.semantic * semantic_score +
                    self.weights.budget_fit * budget_fit +
                    self.weights.price_advantage * price_advantage
                )
                if final_score < min_score_val:
                    continue
                # Enhanced explanation: include tags, category, and payment options
                tags = payload.get("tags", "")
                if isinstance(tags, list):
                    tags = ', '.join(tags)
                explanation = (
                    f"**Why this product?**\n"
                    f"- Query match: {semantic_score*100:.1f}%\n"
                    f"- Price: €{price:.2f} (Budget: €{user_budget:.2f})\n"
                    f"- Category: {payload.get('category', '')}\n"
                    f"- Tags: {tags}\n"
                    f"- Installments: {'Yes' if payload.get('installment_available', False) else 'No'}\n"
                    f"- Composite score: {final_score:.2f}/1.00\n"
                )
                ranked_results.append({
                    "product_id": payload.get("product_id"),
                    "title": payload.get("title", ""),
                    "description": payload.get("description", ""),
                    "price": price,
                    "category": payload.get("category", ""),
                    "brand": payload.get("brand", ""),
                    "rating": payload.get("rating", 0),
                    "semantic_score": round(semantic_score, 4),
                    "budget_fit_score": budget_fit,
                    "price_advantage_score": round(price_advantage, 4),
                    "composite_score": round(final_score, 4),
                    "explanation": explanation,
                    "msrp": payload.get("msrp"),
                    "discount_pct": payload.get("discount_pct"),
                    "installment_available": payload.get("installment_available", False),
                    "max_installments": payload.get("max_installments", 0),
                    "shipping_days": payload.get("shipping_days", 0),
                    "budget_band": payload.get("budget_band", ""),
                })
            except Exception as e:
                logger.warning(f"Failed to rank result: {e}")
                continue
        ranked_results.sort(key=lambda x: x['composite_score'], reverse=True)
        logger.info(f"Ranked {len(ranked_results)} results from {len(search_hits)} hits")
        # Suggest alternatives if results are low
        if len(ranked_results) < 3 and len(search_hits) > 3:
            logger.info("Few results after ranking, consider suggesting alternatives with relaxed constraints.")
        return ranked_results
    
    @staticmethod
    def explain_no_results(query: str, budget: float, filters: Dict[str, Any] = None, *args, **kwargs) -> str:
        """Explain why no results were found and suggest actions.
        
        Args:
            query: User query
            budget: User budget
            filters: Applied filters
            
        Returns:
            Explanation with suggestions
        """
        suggestions = [
            f"No products found for '{query}' within €{budget:.2f} budget.",
            "",
            "**Suggestions:**",
            "1. Try increasing your budget",
            "2. Use different search terms (e.g., more general keywords)",
            "3. Remove category filters if applied",
            "4. Check if similar products exist in different categories"
        ]
        
        return "\n".join(suggestions)


def rank_and_explain(search_hits: List[Any], user_budget: float, 
                     weights: Optional[ScoringWeights] = None) -> List[Dict[str, Any]]:
    """Convenience function for ranking with explanations (backward compatible).
    
    Args:
        search_hits: Results from vector search
        user_budget: User's budget
        weights: Optional scoring weights
        
    Returns:
        Ranked and explained results
    """
    ranker = Ranker(weights=weights)
    return ranker.rank(search_hits, user_budget)


__all__ = ['Ranker', 'Explainer', 'ScoringWeights', 'rank_and_explain']