import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import config
except ImportError:
    # Fallback if config is not available
    class Config:
        @staticmethod
        def get_path(filename: str) -> str:
            return filename
    config = Config()


class HotelReviewQualityAnalyzer:
    """
    Analyzes hotel review quality and calculates quality scores based on various metrics.
    
    Metrics considered:
    - Review count
    - Reviews with images
    - Long reviews (>150 characters)
    - Average review scores
    """
    
    # Constants for scoring
    LONG_REVIEW_THRESHOLD = 150
    IMAGE_RATIO_THRESHOLD = 0.5
    LONG_REVIEW_RATIO_THRESHOLD = 0.5
    
    # Scoring weights
    DEFAULT_WEIGHTS = {
        'avg_score': 0.2,
        'count_review': 0.6,
        'image_ratio': 0.1,
        'long_review_ratio': 0.1
    }
    
    DYNAMIC_WEIGHTS_HIGH = {
        'avg_score': 0.3,
        'count_review': 0.5,
        'image_ratio': 0.1,
        'long_review_ratio': 0.1
    }

    def __init__(self, reviews: Optional[List[Dict]] = None):
        """
        Initialize the analyzer.
        
        Args:
            reviews: List of review dictionaries
        """
        self.reviews = reviews or []
        self.stats = defaultdict(self._default_stats_dict)
        self.final_scores = {}

    @staticmethod
    def _default_stats_dict() -> Dict:
        """Default dictionary structure for hotel statistics."""
        return {
            "count_review": 0,
            "count_review_with_image": 0,
            "count_long_review": 0,
            "total_score": 0.0,
            "num_score": 0,
            "avg_score": 0.0
        }

    def process_reviews(self) -> Dict:
        """
        Process reviews and calculate basic statistics.
        
        Returns:
            Dictionary containing statistics for each hotel
        """
        if not self.reviews:
            print("Warning: No reviews to process")
            return dict(self.stats)

        for review in tqdm(self.reviews, desc="Processing reviews"):
            hotel_id = review.get('id')
            if not hotel_id:
                continue
                
            self._process_single_review(hotel_id, review)
        
        return dict(self.stats)

    def _process_single_review(self, hotel_id: str, review: Dict) -> None:
        """
        Process a single review and update statistics.
        
        Args:
            hotel_id: Hotel identifier
            review: Review dictionary
        """
        stats = self.stats[hotel_id]
        stats['count_review'] += 1
        
        # Count reviews with images
        if review.get('review_photo'):
            stats['count_review_with_image'] += 1
            
        # Count long reviews
        review_text = self._extract_review_text(review)
        if len(review_text) > self.LONG_REVIEW_THRESHOLD:
            stats['count_long_review'] += 1
        
        # Process review score
        self._process_review_score(stats, review)

    def _extract_review_text(self, review: Dict) -> str:
        """Extract and combine review text from positive and negative fields."""
        positive = review.get("review_positive", "")
        negative = review.get("review_negative", "")
        return f"{positive} {negative}".strip()

    def _process_review_score(self, stats: Dict, review: Dict) -> None:
        """
        Process review score and update statistics.
        
        Args:
            stats: Hotel statistics dictionary
            review: Review dictionary
        """
        score_str = review.get("review_score")
        if not score_str:
            return
            
        try:
            # Extract numeric score (handles formats like "Score 8.5")
            score = float(score_str.split()[-1].replace(",", "."))
            stats["total_score"] += score
            stats["num_score"] += 1
            stats["avg_score"] = stats["total_score"] / stats["num_score"]
        except (ValueError, IndexError):
            # Skip invalid scores
            pass

    def save_stats(self, filename: str = 'review_quality.json') -> None:
        """
        Save statistics to JSON file.
        
        Args:
            filename: Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dict(self.stats), f, ensure_ascii=False, indent=2)
            print(f"Statistics saved to {filename}")
        except Exception as e:
            print(f"Error saving statistics: {e}")

    def load_stats(self, filename: Optional[str] = None) -> Dict:
        """
        Load statistics from JSON file.
        
        Args:
            filename: Input filename. If None, uses config path
        
        Returns:
            Loaded statistics dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if filename is None:
            filename = config.get_path("review_quality.json")
            
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                loaded_stats = json.load(f)
            
            # Convert to defaultdict
            self.stats = defaultdict(self._default_stats_dict)
            self.stats.update(loaded_stats)
            
            print(f"Statistics loaded from {filename}")
            return dict(self.stats)
            
        except Exception as e:
            print(f"Error loading statistics: {e}")
            raise

    def calculate_statistics(self, hotel_ids: Optional[List[str]] = None) -> Dict:
        """
        Calculate statistical summaries for metrics.
        
        Args:
            hotel_ids: List of hotel IDs to include. If None, uses all hotels
        
        Returns:
            Dictionary containing statistical summaries
        """
        # Filter stats if hotel_ids provided
        if hotel_ids is not None:
            filtered_stats = {hid: self.stats[hid] for hid in hotel_ids if hid in self.stats}
        else:
            filtered_stats = self.stats
        
        if not filtered_stats:
            return {}
        
        # Extract metric values
        metrics = self._extract_metric_values(filtered_stats)
        
        # Calculate statistics for each metric
        return {
            metric: self._compute_metric_stats(values) 
            for metric, values in metrics.items()
        }

    def _extract_metric_values(self, stats_dict: Dict) -> Dict[str, List]:
        """Extract metric values from statistics dictionary."""
        metrics = {
            'count_review': [],
            'count_review_with_image': [],
            'count_long_review': [],
            'avg_score': []
        }
        
        for data in stats_dict.values():
            metrics['count_review'].append(data['count_review'])
            metrics['count_review_with_image'].append(data['count_review_with_image'])
            metrics['count_long_review'].append(data['count_long_review'])
            metrics['avg_score'].append(data['avg_score'])
        
        return metrics

    def _compute_metric_stats(self, values: List[float]) -> Dict:
        """Compute statistical summary for a list of values."""
        if not values:
            return {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)) if len(values) > 1 else 0,
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    def calculate_local_scores(self, hotel_ids: List[str], print_warnings: bool = False) -> Dict[str, float]:
        """
        Calculate quality scores for specified hotels using local dataset statistics.
        Uses dynamic weighting based on review count.
        
        Args:
            hotel_ids: List of hotel IDs to score
            print_warnings: Whether to print warnings for invalid IDs
        
        Returns:
            Dictionary mapping hotel_id to quality score (0-1 scale)
        """
        valid_ids = [hid for hid in hotel_ids if hid in self.stats]
        
        if not valid_ids:
            if print_warnings:
                print("Warning: No valid hotel IDs provided")
            return {}

        # Calculate statistics for normalization
        local_stats = self.calculate_statistics(hotel_ids=valid_ids)
        if not local_stats:
            return {}

        review_count_threshold = local_stats['count_review']['mean']
        
        scores = {}
        for hotel_id in hotel_ids:
            if hotel_id not in self.stats:
                if print_warnings:
                    print(f"Warning: Hotel ID {hotel_id} not found in stats")
                continue
            
            score = self._calculate_single_score(
                hotel_id, local_stats, review_count_threshold
            )
            scores[hotel_id] = round(score, 3)
        
        return scores

    def calculate_global_scores(self) -> Dict[str, float]:
        """
        Calculate quality scores for all hotels using global dataset statistics.
        
        Returns:
            Dictionary mapping hotel_id to quality score (0-10 scale)
        """
        if not self.stats:
            print("Warning: No statistics available")
            return {}

        global_stats = self.calculate_statistics()
        if not global_stats:
            return {}

        self.final_scores = {}
        for hotel_id in self.stats:
            score = self._calculate_single_score(hotel_id, global_stats) * 10.0
            self.final_scores[hotel_id] = round(score, 2)
        
        return self.final_scores

    def _calculate_single_score(self, hotel_id: str, stats_info: Dict, 
                              review_threshold: Optional[float] = None) -> float:
        """
        Calculate quality score for a single hotel.
        
        Args:
            hotel_id: Hotel identifier
            stats_info: Statistical information for normalization
            review_threshold: Threshold for dynamic weighting
        
        Returns:
            Quality score (0-1 scale)
        """
        data = self.stats[hotel_id]
        
        # Normalize metrics
        normalized_metrics = self._normalize_metrics(data, stats_info)
        
        # Determine weights
        weights = self._get_weights(data['count_review'], review_threshold)
        
        # Calculate weighted score
        score = (
            weights['avg_score'] * normalized_metrics['avg_score'] +
            weights['count_review'] * normalized_metrics['count_review'] +
            weights['image_ratio'] * normalized_metrics['image_ratio'] +
            weights['long_review_ratio'] * normalized_metrics['long_review_ratio']
        )
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]

    def _normalize_metrics(self, data: Dict, stats_info: Dict) -> Dict[str, float]:
        """Normalize hotel metrics using statistical information."""
        avg_score_range = stats_info['avg_score']['max'] - stats_info['avg_score']['min']
        
        # Normalize average score (linear)
        if avg_score_range > 0:
            norm_avg_score = ((data['avg_score'] - stats_info['avg_score']['min']) / 
                            avg_score_range)
        else:
            norm_avg_score = 0.0
        
        # Normalize review count (logarithmic to reduce impact of outliers)
        max_log_review = math.log1p(stats_info['count_review']['max'])
        norm_count_review = (math.log1p(data['count_review']) / max_log_review 
                           if data['count_review'] > 0 else 0.0)
        
        # Normalize ratios with thresholds
        count_review = data['count_review']
        
        image_ratio = (data['count_review_with_image'] / count_review 
                      if count_review > 0 else 0.0)
        norm_image_ratio = min(image_ratio / self.IMAGE_RATIO_THRESHOLD, 1.0)
        
        long_review_ratio = (data['count_long_review'] / count_review 
                           if count_review > 0 else 0.0)
        norm_long_review_ratio = min(long_review_ratio / self.LONG_REVIEW_RATIO_THRESHOLD, 1.0)
        
        return {
            'avg_score': norm_avg_score,
            'count_review': norm_count_review,
            'image_ratio': norm_image_ratio,
            'long_review_ratio': norm_long_review_ratio
        }

    def _get_weights(self, review_count: int, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Get scoring weights, potentially using dynamic weighting.
        
        Args:
            review_count: Number of reviews for the hotel
            threshold: Threshold for dynamic weighting
        
        Returns:
            Dictionary of weights
        """
        if threshold is not None and review_count > threshold:
            return self.DYNAMIC_WEIGHTS_HIGH.copy()
        else:
            return self.DEFAULT_WEIGHTS.copy()

    def get_top_hotels(self, n: int = 10, score_type: str = 'global') -> List[Tuple[str, float]]:
        """
        Get top N hotels by quality score.
        
        Args:
            n: Number of hotels to return
            score_type: 'global' or 'local'
        
        Returns:
            List of (hotel_id, score) tuples sorted by score descending
        """
        if score_type == 'global':
            if not self.final_scores:
                self.calculate_global_scores()
            scores = self.final_scores
        else:
            # For local scores, calculate for all available hotels
            hotel_ids = list(self.stats.keys())
            scores = self.calculate_local_scores(hotel_ids)
        
        if not scores:
            return []
        
        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:n]

    def create_visualizations(self, save_plots: bool = True, output_dir: str = '.') -> None:
        """
        Create all visualization plots.
        
        Args:
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots
        """
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        self.plot_distributions(
            save_path=os.path.join(output_dir, 'distributions.png') if save_plots else None
        )
        self.plot_scatter_analysis(
            save_path=os.path.join(output_dir, 'scatter_analysis.png') if save_plots else None
        )
        self.plot_score_distribution(
            save_path=os.path.join(output_dir, 'score_distribution.png') if save_plots else None
        )

    def plot_distributions(self, save_path: Optional[str] = None) -> None:
        """Plot distributions of key metrics."""
        if not self.stats:
            print("No statistics available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Distribution of Hotel Review Metrics', fontsize=16)
        
        metrics = self._extract_metric_values(dict(self.stats))
        colors = ['skyblue', 'lightgreen', 'salmon', 'lightcoral']
        titles = [
            'Review Count Distribution',
            'Reviews with Images Distribution', 
            'Long Reviews Distribution',
            'Average Score Distribution'
        ]
        
        for i, (metric, values) in enumerate(metrics.items()):
            row, col = i // 2, i % 2
            axes[row, col].hist(values, bins=20, color=colors[i], 
                              edgecolor='black', alpha=0.7)
            axes[row, col].set_title(titles[i])
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distributions plot saved to {save_path}")
        
        plt.show()

    def plot_scatter_analysis(self, save_path: Optional[str] = None) -> None:
        """Plot scatter analysis of review metrics."""
        if not self.stats:
            print("No statistics available for plotting")
            return
        
        plt.figure(figsize=(10, 6))
        
        metrics = self._extract_metric_values(dict(self.stats))
        count_reviews = metrics['count_review']
        avg_scores = metrics['avg_score']
        long_reviews = metrics['count_long_review']
        
        # Scale bubble sizes
        max_long = max(long_reviews) if long_reviews else 1
        sizes = [(lr / max_long) * 100 + 10 for lr in long_reviews]  # Min size 10
        
        scatter = plt.scatter(count_reviews, avg_scores, s=sizes, alpha=0.6, 
                            c=long_reviews, cmap='viridis', edgecolor='black', linewidth=0.5)
        
        plt.colorbar(scatter, label='Long Reviews Count')
        plt.title('Hotel Quality Analysis: Avg Score vs Review Count\n(Bubble size = Long Reviews)')
        plt.xlabel('Total Review Count')
        plt.ylabel('Average Review Score')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatter analysis plot saved to {save_path}")
        
        plt.show()

    def plot_score_distribution(self, save_path: Optional[str] = None) -> None:
        """Plot distribution of calculated quality scores."""
        if not self.final_scores:
            self.calculate_global_scores()
        
        if not self.final_scores:
            print("No final scores available for plotting")
            return
        
        plt.figure(figsize=(8, 6))
        
        scores = list(self.final_scores.values())
        plt.hist(scores, bins=20, color='purple', alpha=0.7, 
                edgecolor='black')
        plt.title('Distribution of Hotel Quality Scores')
        plt.xlabel('Quality Score (0-10)')
        plt.ylabel('Number of Hotels')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        plt.axvline(mean_score, color='red', linestyle='--', 
                   label=f'Mean: {mean_score:.2f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Score distribution plot saved to {save_path}")
        
        plt.show()

    def generate_report(self, output_file: str = 'hotel_quality_report.txt') -> None:
        """
        Generate a comprehensive text report of the analysis.
        
        Args:
            output_file: Path to output report file
        """
        if not self.stats:
            print("No statistics available for report generation")
            return
        
        overall_stats = self.calculate_statistics()
        top_hotels = self.get_top_hotels(10)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("HOTEL REVIEW QUALITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL DATASET STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total hotels analyzed: {len(self.stats)}\n\n")
            
            for metric, stats in overall_stats.items():
                f.write(f"{metric.replace('_', ' ').title()}:\n")
                f.write(f"  Mean: {stats['mean']:.2f}\n")
                f.write(f"  Std:  {stats['std']:.2f}\n")
                f.write(f"  Min:  {stats['min']:.2f}\n")
                f.write(f"  Max:  {stats['max']:.2f}\n\n")
            
            # Top hotels
            f.write("TOP 10 HOTELS BY QUALITY SCORE\n")
            f.write("-" * 30 + "\n")
            for i, (hotel_id, score) in enumerate(top_hotels, 1):
                f.write(f"{i:2d}. {hotel_id}: {score:.2f}\n")
        
        print(f"Report generated: {output_file}")


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    analyzer = HotelReviewQualityAnalyzer()
    
    # Load existing stats or process new reviews
    try:
        analyzer.load_stats()
        print("Statistics loaded successfully")
    except FileNotFoundError:
        print("No existing statistics found. Please process reviews first.")
    
    # Generate visualizations
    if analyzer.stats:
        analyzer.create_visualizations(save_plots=True)
        analyzer.generate_report()
        
        # Print top hotels
        top_hotels = analyzer.get_top_hotels(5)
        print("\nTop 5 Hotels:")
        for i, (hotel_id, score) in enumerate(top_hotels, 1):
            print(f"{i}. {hotel_id}: {score:.2f}")