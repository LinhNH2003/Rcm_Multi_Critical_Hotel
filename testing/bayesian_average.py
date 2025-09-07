"""
Bayesian Analysis System for Hotel Rating Evaluation

This module provides comprehensive tools for analyzing and visualizing
Bayesian averaging effects on hotel ratings with different parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import seaborn as sns

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


@dataclass
class BayesianConfig:
    """Configuration for Bayesian analysis parameters."""
    c_values: List[float] = None
    sample_ratings: List[float] = None
    review_range: Tuple[int, int, int] = (10, 1001, 10)  # start, stop, step
    figure_size: Tuple[int, int] = (15, 10)
    
    def __post_init__(self):
        if self.c_values is None:
            self.c_values = [10, 26, 66, 184, 188]
        if self.sample_ratings is None:
            self.sample_ratings = [6, 7, 9, 10]


class HotelDataLoader:
    """Handles loading and preprocessing hotel review data."""
    
    @staticmethod
    def load_from_json(file_path: Union[str, Path]) -> Dict:
        """
        Load hotel data from JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary containing hotel data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Hotel data file not found: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file_path}")
    
    @staticmethod
    def validate_data(data: Dict) -> bool:
        """
        Validate hotel data structure.
        
        Args:
            data: Hotel data dictionary
            
        Returns:
            True if data is valid, raises exception otherwise
        """
        required_fields = ['avg_score', 'count_review']
        
        for hotel_id, info in data.items():
            for field in required_fields:
                if field not in info:
                    raise ValueError(f"Missing field '{field}' in hotel {hotel_id}")
                if not isinstance(info[field], (int, float)):
                    raise ValueError(f"Field '{field}' must be numeric in hotel {hotel_id}")
        
        return True


class BayesianAnalyzer:
    """Core class for Bayesian analysis of hotel ratings."""
    
    def __init__(self, hotel_data: Dict, config: Optional[BayesianConfig] = None):
        """
        Initialize analyzer with hotel data.
        
        Args:
            hotel_data: Dictionary containing hotel review data
            config: Configuration object for analysis parameters
        """
        HotelDataLoader.validate_data(hotel_data)
        self.hotel_data = hotel_data
        self.config = config or BayesianConfig()
        self.global_mean = self._calculate_global_mean()
        
    def _calculate_global_mean(self) -> float:
        """Calculate global weighted average rating."""
        total_rating = sum(
            hotel["avg_score"] * hotel["count_review"] 
            for hotel in self.hotel_data.values()
        )
        total_reviews = sum(
            hotel["count_review"] 
            for hotel in self.hotel_data.values()
        )
        
        if total_reviews == 0:
            raise ValueError("No reviews found in the dataset")
        
        return total_rating / total_reviews
    
    @staticmethod
    def bayesian_average(avg_rating: float, num_reviews: int, c: float, m: float) -> float:
        """
        Calculate Bayesian average score.
        
        Args:
            avg_rating: Hotel's average rating
            num_reviews: Number of reviews
            c: Bayesian confidence parameter
            m: Global mean rating
            
        Returns:
            Bayesian adjusted score
        """
        if num_reviews + c == 0:
            return m
        return (num_reviews * avg_rating + c * m) / (num_reviews + c)
    
    def analyze_sample_hotels(self, hotel_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze Bayesian scores for sample hotels.
        
        Args:
            hotel_ids: List of specific hotel IDs to analyze (uses all if None)
            
        Returns:
            DataFrame with analysis results
        """
        if hotel_ids is None:
            hotel_ids = list(self.hotel_data.keys())
        
        results = []
        
        for hotel_id in hotel_ids:
            if hotel_id not in self.hotel_data:
                continue
                
            info = self.hotel_data[hotel_id]
            avg_score = info['avg_score']
            count_review = info['count_review']
            
            for c in self.config.c_values:
                bayesian_score = self.bayesian_average(avg_score, count_review, c, self.global_mean)
                
                results.append({
                    'hotel_id': hotel_id,
                    'original_rating': avg_score,
                    'review_count': count_review,
                    'c_value': c,
                    'bayesian_score': bayesian_score,
                    'difference': bayesian_score - avg_score
                })
        
        return pd.DataFrame(results)
    
    def generate_convergence_data(self) -> pd.DataFrame:
        """
        Generate data showing how Bayesian scores converge with review count.
        
        Returns:
            DataFrame with convergence analysis data
        """
        start, stop, step = self.config.review_range
        n_values = np.arange(start, stop, step)
        
        results = []
        
        for avg_rating in self.config.sample_ratings:
            for c in self.config.c_values:
                for n in n_values:
                    bayesian_score = self.bayesian_average(avg_rating, n, c, self.global_mean)
                    
                    results.append({
                        'sample_rating': avg_rating,
                        'c_value': c,
                        'review_count': n,
                        'bayesian_score': bayesian_score,
                        'convergence_ratio': abs(bayesian_score - avg_rating) / abs(self.global_mean - avg_rating)
                        if avg_rating != self.global_mean else 0
                    })
        
        return pd.DataFrame(results)


class BayesianVisualizer:
    """Handles visualization of Bayesian analysis results."""
    
    def __init__(self, analyzer: BayesianAnalyzer):
        """Initialize visualizer with analyzer instance."""
        self.analyzer = analyzer
        self.config = analyzer.config
        
    def plot_convergence_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Plot convergence analysis for different sample ratings and C values.
        
        Args:
            save_path: Optional path to save the plot
        """
        convergence_data = self.analyzer.generate_convergence_data()
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        axes = axes.flatten()
        
        for i, sample_rating in enumerate(self.config.sample_ratings):
            if i >= len(axes):
                break
                
            ax = axes[i]
            subset = convergence_data[convergence_data['sample_rating'] == sample_rating]
            
            for c in self.config.c_values:
                c_data = subset[subset['c_value'] == c]
                ax.plot(c_data['review_count'], c_data['bayesian_score'], 
                       label=f'C = {c}', linewidth=2)
            
            # Reference lines
            ax.axhline(y=sample_rating, color='red', linestyle='--', 
                      label=f'Điểm thực (r̄ = {sample_rating})', alpha=0.7)
            ax.axhline(y=self.analyzer.global_mean, color='green', linestyle='--', 
                      label=f'Điểm TB toàn cục (m = {self.analyzer.global_mean:.2f})', alpha=0.7)
            
            ax.set_xlabel('Số lượng review (n)')
            ax.set_ylabel('Điểm Bayesian')
            ax.set_title(f'Điểm mẫu r̄ = {sample_rating}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_single_convergence(self, sample_rating: float = 8.5, save_path: Optional[str] = None) -> None:
        """
        Plot convergence for a single sample rating.
        
        Args:
            sample_rating: The sample rating to analyze
            save_path: Optional path to save the plot
        """
        start, stop, step = self.config.review_range
        n_values = np.arange(start, stop, step)
        
        plt.figure(figsize=(12, 8))
        
        for c in self.config.c_values:
            scores = [self.analyzer.bayesian_average(sample_rating, n, c, self.analyzer.global_mean) 
                     for n in n_values]
            plt.plot(n_values, scores, label=f'C = {c}', linewidth=2)
        
        plt.axhline(y=sample_rating, color='red', linestyle='--', 
                   label=f'Điểm thực (r̄ = {sample_rating})', linewidth=2)
        plt.axhline(y=self.analyzer.global_mean, color='green', linestyle='--', 
                   label=f'Điểm TB toàn cục (m = {self.analyzer.global_mean:.2f})', linewidth=2)
        
        plt.xlabel('Số lượng review (n)', fontsize=12)
        plt.ylabel('Điểm Bayesian', fontsize=12)
        plt.title('Ảnh hưởng của C đến điểm Bayesian khi số review tăng', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_hotel_comparison(self, hotel_ids: Optional[List[str]] = None, save_path: Optional[str] = None) -> None:
        """
        Plot comparison of original vs Bayesian scores for hotels.
        
        Args:
            hotel_ids: List of hotel IDs to compare
            save_path: Optional path to save the plot
        """
        sample_data = self.analyzer.analyze_sample_hotels(hotel_ids)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Score comparison
        pivot_scores = sample_data.pivot_table(
            values='bayesian_score', 
            index='hotel_id', 
            columns='c_value'
        )
        
        pivot_scores.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Bayesian Scores by C Value')
        axes[0].set_ylabel('Bayesian Score')
        axes[0].legend(title='C Value')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Difference from original
        pivot_diff = sample_data.pivot_table(
            values='difference', 
            index='hotel_id', 
            columns='c_value'
        )
        
        pivot_diff.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Difference from Original Rating')
        axes[1].set_ylabel('Score Difference')
        axes[1].legend(title='C Value')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class BayesianReporter:
    """Generates reports and summaries of Bayesian analysis."""
    
    def __init__(self, analyzer: BayesianAnalyzer):
        """Initialize reporter with analyzer instance."""
        self.analyzer = analyzer
    
    def print_global_statistics(self) -> None:
        """Print global dataset statistics."""
        total_hotels = len(self.analyzer.hotel_data)
        total_reviews = sum(hotel['count_review'] for hotel in self.analyzer.hotel_data.values())
        avg_reviews_per_hotel = total_reviews / total_hotels if total_hotels > 0 else 0
        
        print("=" * 60)
        print("THỐNG KÊ TOÀN CỤC")
        print("=" * 60)
        print(f"Số lượng khách sạn: {total_hotels:,}")
        print(f"Tổng số reviews: {total_reviews:,}")
        print(f"Trung bình reviews/khách sạn: {avg_reviews_per_hotel:.1f}")
        print(f"Điểm trung bình toàn cục (m): {self.analyzer.global_mean:.4f}")
        print()
    
    def print_sample_analysis(self, hotel_ids: Optional[List[str]] = None, max_hotels: int = 5) -> None:
        """
        Print detailed analysis for sample hotels.
        
        Args:
            hotel_ids: List of hotel IDs to analyze
            max_hotels: Maximum number of hotels to display
        """
        if hotel_ids is None:
            hotel_ids = list(self.analyzer.hotel_data.keys())[:max_hotels]
        
        sample_data = self.analyzer.analyze_sample_hotels(hotel_ids)
        
        print("PHÂN TÍCH MẪU KHÁCH SẠN")
        print("=" * 60)
        
        for hotel_id in hotel_ids:
            hotel_data = sample_data[sample_data['hotel_id'] == hotel_id]
            if hotel_data.empty:
                continue
                
            info = self.analyzer.hotel_data[hotel_id]
            print(f"\nKhách sạn {hotel_id}:")
            print(f"  Điểm trung bình: {info['avg_score']:.4f}")
            print(f"  Số review: {info['count_review']:,}")
            print("  Điểm Bayesian:")
            
            for _, row in hotel_data.iterrows():
                print(f"    C = {row['c_value']:3.0f} → {row['bayesian_score']:.4f} "
                      f"(Δ = {row['difference']:+.4f})")
    
    def print_convergence_summary(self, specific_n_values: Optional[List[int]] = None) -> None:
        """
        Print convergence summary for specific review counts.
        
        Args:
            specific_n_values: List of specific review counts to analyze
        """
        if specific_n_values is None:
            specific_n_values = [10, 50, 100, 200, 500, 1000]
        
        print("PHÂN TÍCH HỘI TỤ")
        print("=" * 60)
        print(f"Điểm trung bình toàn cục (m): {self.analyzer.global_mean:.4f}\n")
        
        for sample_rating in self.analyzer.config.sample_ratings:
            print(f"Điểm mẫu r̄ = {sample_rating}:")
            
            for c in self.analyzer.config.c_values:
                print(f"\n  C = {c}:")
                for n in specific_n_values:
                    score = self.analyzer.bayesian_average(
                        sample_rating, n, c, self.analyzer.global_mean
                    )
                    print(f"    n = {n:4d} → Điểm Bayesian: {score:.4f}")
            print()


# Convenience function for quick analysis
def run_bayesian_analysis(
    data_path: str,
    config: Optional[BayesianConfig] = None,
    save_plots: bool = False,
    plot_dir: str = "./plots"
) -> BayesianAnalyzer:
    """
    Run complete Bayesian analysis pipeline.
    
    Args:
        data_path: Path to hotel data JSON file
        config: Configuration object
        save_plots: Whether to save plots
        plot_dir: Directory to save plots
        
    Returns:
        Configured BayesianAnalyzer instance
    """
    # Load and validate data
    hotel_data = HotelDataLoader.load_from_json(data_path)
    
    # Initialize analyzer
    analyzer = BayesianAnalyzer(hotel_data, config)
    
    # Generate reports
    reporter = BayesianReporter(analyzer)
    reporter.print_global_statistics()
    reporter.print_sample_analysis()
    reporter.print_convergence_summary()
    
    # Generate visualizations
    visualizer = BayesianVisualizer(analyzer)
    
    if save_plots:
        Path(plot_dir).mkdir(exist_ok=True)
        visualizer.plot_single_convergence(save_path=f"{plot_dir}/single_convergence.png")
        visualizer.plot_convergence_analysis(save_path=f"{plot_dir}/convergence_analysis.png")
        visualizer.plot_hotel_comparison(save_path=f"{plot_dir}/hotel_comparison.png")
    else:
        visualizer.plot_single_convergence()
        visualizer.plot_convergence_analysis()
        visualizer.plot_hotel_comparison()
    
    return analyzer


# Example usage
if __name__ == "__main__":
    # Example usage with custom configuration
    custom_config = BayesianConfig(
        c_values=[10, 50, 100, 200, 300],
        sample_ratings=[6, 7, 8, 9, 10],
        review_range=(5, 2001, 20),
        figure_size=(18, 12)
    )
    
    # For demonstration with mock data
    mock_data = {
        "hotel_1": {"avg_score": 8.5, "count_review": 150},
        "hotel_2": {"avg_score": 9.2, "count_review": 89},
        "hotel_3": {"avg_score": 7.8, "count_review": 245},
        "hotel_4": {"avg_score": 9.8, "count_review": 23},
        "hotel_5": {"avg_score": 6.5, "count_review": 340}
    }
    
    # Run analysis with mock data
    analyzer = BayesianAnalyzer(mock_data, custom_config)
    visualizer = BayesianVisualizer(analyzer)
    reporter = BayesianReporter(analyzer)
    
    # Generate reports and visualizations
    reporter.print_global_statistics()
    reporter.print_sample_analysis()
    visualizer.plot_single_convergence(8.5)
    