"""
Hotel Scoring and Recommendation System

This module provides functionality to calculate weighted Bayesian scores and final scores
for hotels based on review data and quality metrics.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoringMethod(Enum):
    """Enumeration for different scoring methods."""
    EQUAL_WEIGHTS = "equal_weights"
    WEIGHTED_RATIO = "weighted_ratio"
    REVIEW_COUNT_BASED = "review_count_based"


@dataclass
class ScoringConfig:
    """Configuration class for scoring parameters."""
    default_weight: float = 1/16
    bayesian_constant: float = 100
    quality_weight: float = 0.8
    default_quality: float = 0.5
    global_score: float = 0.5
    quality_threshold: float = 0.0


class HotelCriteriaConstants:
    """Constants for hotel evaluation criteria."""
    
    ALL_CRITERIA = [
        'location', 'room', 'cleanliness', 'comfort', 'service',
        'staff', 'wifi', 'food', 'bathroom', 'parking',
        'value', 'facilities', 'air conditioning', 'view',
        'environment', 'security'
    ]


class WeightAllocator:
    """Handles weight allocation for hotel evaluation criteria."""
    
    @staticmethod
    def allocate_weights(
        selected_weights: Optional[Dict[str, float]] = None,
        total_selected_weight: float = 0.0,
        criteria_list: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Allocate weights for evaluation criteria.
        
        Args:
            selected_weights: Dictionary mapping criteria to relative weights
            total_selected_weight: Total weight to allocate to selected criteria
            criteria_list: List of all criteria (uses default if None)
            
        Returns:
            Dictionary containing allocated weights for all criteria
        """
        if criteria_list is None:
            criteria_list = HotelCriteriaConstants.ALL_CRITERIA.copy()
        
        num_criteria = len(criteria_list)
        
        # Case 1: No specific weights provided - equal distribution
        if selected_weights is None:
            equal_weight = 1.0 / num_criteria
            return {key: equal_weight for key in criteria_list}
        
        # Case 2: Calculate weights based on ratios
        if total_selected_weight == 0:
            total_selected_weight = sum(selected_weights.values()) / 10
        
        # Validate total_selected_weight
        if total_selected_weight > 1.0:
            logger.warning(f"Total selected weight {total_selected_weight} > 1.0, clamping to 1.0")
            total_selected_weight = 1.0
        
        # Calculate ratio-based allocation
        total_ratio = sum(selected_weights.values())
        if total_ratio == 0:
            return WeightAllocator.allocate_weights(None, 0, criteria_list)
        
        per_unit_weight = total_selected_weight / total_ratio
        
        # Allocate weights based on ratios
        result = {}
        total_allocated = 0
        
        for key in criteria_list:
            if key in selected_weights:
                weight = selected_weights[key] * per_unit_weight
                result[key] = weight
                total_allocated += weight
            else:
                result[key] = 0
        
        # Distribute remaining weight equally among all criteria
        remainder = max(0, 1.0 - total_allocated)
        remainder_per_key = remainder / num_criteria
        
        for key in result:
            result[key] += remainder_per_key
        
        return result


class BayesianScoreCalculator:
    """Calculates Bayesian-adjusted scores for hotels."""
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        """Initialize calculator with configuration."""
        self.config = config or ScoringConfig()
    
    def _extract_hotel_data(self, hotel_data: Dict) -> Tuple[Dict[str, float], Dict[str, int], int, float]:
        """Extract relevant data from hotel dictionary."""
        criteria_summary = hotel_data['summary']['criteria_summary']
        review_count = hotel_data['summary']['review_count']
        avg_rating = hotel_data['summary']['avg_rating']
        
        positive_rates = {}
        total_reviews = {}
        
        for criterion, ratings in criteria_summary.items():
            positive_rates[criterion] = ratings['Positive']
            total_reviews[criterion] = ratings['total']
        
        return positive_rates, total_reviews, review_count, avg_rating
    
    def _calculate_weights(
        self, 
        user_weights: Optional[Dict[str, float]], 
        criteria_summary: Dict,
        review_count: int
    ) -> Dict[str, float]:
        """Calculate weights based on user preferences or review counts."""
        if user_weights is None:
            # Weight by review count ratio
            weights = {}
            for criterion, ratings in criteria_summary.items():
                total_criterion_reviews = ratings['total']
                weights[criterion] = total_criterion_reviews / (review_count * 10)
            return weights
        else:
            # Use provided weights with defaults
            weights = {}
            for criterion in criteria_summary:
                weights[criterion] = user_weights.get(criterion, self.config.default_weight)
            return weights
    
    def _calculate_global_positive_rate(self, result_data: Dict) -> float:
        """Calculate global average positive rate across all hotels and criteria."""
        total_positive_reviews = 0
        total_reviews = 0
        
        for hotel_data in result_data.values():
            criteria_summary = hotel_data['summary']['criteria_summary']
            for criterion, ratings in criteria_summary.items():
                positive_reviews = ratings['Positive'] * ratings['total']
                total_positive_reviews += positive_reviews
                total_reviews += ratings['total']
        
        return total_positive_reviews / total_reviews if total_reviews > 0 else 0
    
    def calculate_scores(
        self,
        hotel_data: Dict,
        user_weights: Optional[Dict[str, float]] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Calculate weighted Bayesian scores for hotels.
        
        Args:
            hotel_data: Dictionary containing hotel data {hotel_id: {'summary': {...}}}
            user_weights: Dictionary containing user-specified weights {criterion: weight}
            verbose: Whether to print process information
            
        Returns:
            DataFrame with hotel scores and metadata
        """
        results = []
        
        # Calculate global positive rate
        global_mu = self._calculate_global_positive_rate(hotel_data)
        if verbose:
            logger.info(f"Global average positive rate (mu): {global_mu:.4f}")
        
        # Process each hotel
        for hotel_id, data in hotel_data.items():
            positive_rates, total_reviews, review_count, avg_rating = self._extract_hotel_data(data)
            criteria_summary = data['summary']['criteria_summary']
            
            # Calculate weights
            weights = self._calculate_weights(user_weights, criteria_summary, review_count)
            
            # Calculate weighted score
            weighted_score = sum(
                weights[criterion] * positive_rates[criterion]
                for criterion in criteria_summary
            )
            
            # Apply Bayesian adjustment
            bayesian_score = (
                (review_count * weighted_score + self.config.bayesian_constant * global_mu) /
                (review_count + self.config.bayesian_constant)
            )
            
            results.append({
                'hotel_id': hotel_id,
                'review_count': review_count,
                'avg_rating': avg_rating,
                'weighted_score': weighted_score,
                'bayesian_score': bayesian_score,
                'positive_rates': positive_rates,
                'total_reviews': total_reviews,
                'weights': weights
            })
        
        return pd.DataFrame(results)


class FinalScoreCalculator:
    """Calculates final scores combining recommendation and quality scores."""
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        """Initialize calculator with configuration."""
        self.config = config or ScoringConfig()
    
    def _compute_final_score(
        self, 
        recommend_score: float, 
        quality_score: float
    ) -> float:
        """
        Compute final score for a single hotel using Damped Multiplicative Combination.
        
        Args:
            recommend_score: Bayesian-adjusted recommendation score
            quality_score: Quality score from review analysis
            
        Returns:
            Final combined score
        """
        # Case 1: No recommendation score available
        if pd.isna(recommend_score):
            return self.config.global_score * self.config.default_quality
        
        # Case 2: No quality score available
        if pd.isna(quality_score):
            return recommend_score * (1 - self.config.quality_weight) * self.config.default_quality
        
        # Case 3: Quality score below threshold
        if quality_score < self.config.quality_threshold:
            return self.config.global_score * self.config.default_quality
        
        # Case 4: Quality score is zero
        if quality_score == 0:
            return recommend_score * (1 - self.config.quality_weight) * self.config.default_quality
        
        # Standard case: Damped Multiplicative Combination
        return recommend_score * (
            self.config.quality_weight * quality_score + 
            (1 - self.config.quality_weight) * self.config.default_quality
        )
    
    def calculate_final_scores(
        self,
        recommend_scores: Dict[str, float],
        quality_scores: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Calculate final scores for hotels.
        
        Args:
            recommend_scores: Dictionary mapping hotel_id to recommendation score
            quality_scores: Dictionary mapping hotel_id to quality score
            
        Returns:
            DataFrame with final scores, sorted by score descending
        """
        # Get all unique hotel IDs
        all_hotel_ids = list(set(recommend_scores.keys()) | set(quality_scores.keys()))
        
        # Create results list
        results = []
        for hotel_id in all_hotel_ids:
            rec_score = recommend_scores.get(hotel_id, np.nan)
            qual_score = quality_scores.get(hotel_id, np.nan)
            final_score = self._compute_final_score(rec_score, qual_score)
            
            results.append({
                'hotel_id': hotel_id,
                'recommend_score': rec_score,
                'quality_score': qual_score,
                'final_score': final_score
            })
        
        # Create and sort DataFrame
        df = pd.DataFrame(results)
        return df.sort_values(by='final_score', ascending=False).reset_index(drop=True)


# Convenience functions for backward compatibility
def allocate_weights_with_ratios(
    selected_weights: Optional[Dict[str, float]] = None,
    total_selected_weight: float = 0.0
) -> Dict[str, float]:
    """Legacy wrapper for weight allocation."""
    return WeightAllocator.allocate_weights(selected_weights, total_selected_weight)


def calculate_weighted_bayesian_score(
    result_groupby_id: Dict,
    user_weights: Optional[Dict[str, float]] = None,
    default_weight: float = 1/16,
    C: float = 100,
    print_process: bool = False
) -> pd.DataFrame:
    """Legacy wrapper for Bayesian score calculation."""
    config = ScoringConfig(default_weight=default_weight, bayesian_constant=C)
    calculator = BayesianScoreCalculator(config)
    return calculator.calculate_scores(result_groupby_id, user_weights, print_process)


def calculate_final_score_from_reviews(
    score_review: Dict[str, float],
    score_review_quality: Dict[str, float],
    w: float = 0.8,
    q_default: float = 0.5,
    s_global: float = 0.5,
    threshold: float = 0.0
) -> pd.DataFrame:
    """Legacy wrapper for final score calculation."""
    config = ScoringConfig(
        quality_weight=w,
        default_quality=q_default,
        global_score=s_global,
        quality_threshold=threshold
    )
    calculator = FinalScoreCalculator(config)
    return calculator.calculate_final_scores(score_review, score_review_quality)


# Example usage
if __name__ == "__main__":
    # Example of using the refactored classes
    
    # 1. Weight allocation example
    selected_weights = {'location': 2, 'room': 1, 'service': 3}
    weights = WeightAllocator.allocate_weights(selected_weights, 0.7)
    print("Allocated weights:", weights)
    
    # 2. Bayesian scoring example (requires actual hotel data)
    # config = ScoringConfig(bayesian_constant=150, default_weight=0.05)
    # calculator = BayesianScoreCalculator(config)
    # scores_df = calculator.calculate_scores(hotel_data, user_weights)
    
    # 3. Final scoring example
    recommend_scores = {'hotel_1': 0.85, 'hotel_2': 0.92}
    quality_scores = {'hotel_1': 0.75, 'hotel_2': 0.68}
    
    final_calculator = FinalScoreCalculator()
    final_scores = final_calculator.calculate_final_scores(recommend_scores, quality_scores)
    print("\nFinal scores:")
    print(final_scores)