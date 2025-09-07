"""
Business logic services for the hotel recommendation system.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import json
import os
from pathlib import Path

from config.settings import settings
from config.constants import *
from schemas.models import (
    HotelQuery, HotelData, RecommendationResult, 
    ScoringResult, FinalRecommendation, HotelInfo
)


@dataclass
class RecommendationConfig:
    """Configuration for recommendation algorithms."""
    weights: Dict[str, float] = None
    similarity_threshold: float = 0.90
    normalization_factor_base: float = 10.0
    max_results: int = 10
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'amenity': 0.3,
                'location': 0.25,
                'price': 0.2,
                'rating': 0.15,
                'policy': 0.1
            }


class DataService:
    """Service for data loading and management."""
    
    def __init__(self):
        self.data_dir = settings.paths.data_dir
    
    def load_hotel_data(self, filename: str) -> Dict[str, Any]:
        """Load hotel data from JSON file."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Hotel data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_hotel_data(self, data: Dict[str, Any], filename: str) -> None:
        """Save hotel data to JSON file."""
        file_path = self.data_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_review_quality(self, filename: str = "review_quality.json") -> Dict[str, Any]:
        """Load review quality data."""
        return self.load_hotel_data(filename)
    
    def load_merged_clusters(self, filename: str = "merged_clusters_2.json") -> Dict[str, Any]:
        """Load merged clusters data."""
        return self.load_hotel_data(filename)
    
    def load_feature_facilities(self, filename: str = "feature_facilities.json") -> Dict[str, Any]:
        """Load feature facilities data."""
        return self.load_hotel_data(filename)
    
    def load_feature_popular_facilities(self, filename: str = "feature_popular_facilities.json") -> Dict[str, Any]:
        """Load feature popular facilities data."""
        return self.load_hotel_data(filename)


class AmenityService:
    """Service for amenity-related operations."""
    
    def __init__(self):
        self.data_service = DataService()
        self.merged_clusters = None
        self.feature_facilities = None
        self.feature_popular_facilities = None
        self._load_data()
    
    def _load_data(self):
        """Load required data files."""
        try:
            self.merged_clusters = self.data_service.load_merged_clusters()
            self.feature_facilities = self.data_service.load_feature_facilities()
            self.feature_popular_facilities = self.data_service.load_feature_popular_facilities()
        except FileNotFoundError as e:
            print(f"Warning: Could not load data files: {e}")
    
    def convert_amenity(self, value: Union[int, str]) -> Union[str, int]:
        """Convert between amenity ID and name."""
        if isinstance(value, int):
            return AMENITIES.get(value, "Không tìm thấy")
        elif isinstance(value, str):
            for amenity_id, amenity_name in AMENITIES.items():
                if amenity_name == value:
                    return amenity_id
            return "Không tìm thấy"
        return "Định dạng không hợp lệ"
    
    def get_main_cluster(self, clusters: List[Tuple], threshold: float = 0.1) -> List:
        """Get main clusters based on similarity scores."""
        if not clusters:
            return []
        
        clusters.sort(key=lambda x: x[1], reverse=True)
        main_cluster = [clusters[0][0]]
        
        if (len(clusters) > 1 and 
            abs(clusters[0][1] - clusters[1][1]) <= threshold * clusters[0][1]):
            main_cluster.append(clusters[1][0])
        
        return main_cluster
    
    def get_facilities_from_cluster(self, main_clusters: List, 
                                   merged_clusters: Dict) -> List[str]:
        """Get facilities from cluster IDs."""
        facilities = []
        for cluster in main_clusters:
            if cluster in merged_clusters:
                facilities.extend(merged_clusters[cluster])
        return list(set(facilities))
    
    def suggest_related_facilities(self, user_input: str) -> Tuple[List, List[str]]:
        """Suggest related facilities in the same cluster."""
        try:
            # This would need the clustering_facilities module
            # cluster = self.get_main_cluster(find_clusters([user_input]), threshold=0.1)
            # facilities_in_cluster = self.get_facilities_from_cluster(
            #     cluster, self.merged_clusters
            # )
            # suggested_facilities = [fac for fac in facilities_in_cluster if fac != user_input]
            # return cluster, suggested_facilities
            return [], []
        except Exception as e:
            print(f"Error in suggest_related_facilities: {e}")
            return [], []


class LocationService:
    """Service for location-related operations."""
    
    def __init__(self):
        self.neighbor_province = self._load_neighbor_provinces()
    
    def _load_neighbor_provinces(self) -> Dict[str, List[str]]:
        """Load neighbor provinces mapping."""
        # This would be loaded from a configuration file
        return {
            "Hà Nội": ["Hưng Yên", "Hà Nam", "Bắc Ninh", "Vĩnh Phúc"],
            "TP Hồ Chí Minh": ["Bình Dương", "Đồng Nai", "Long An", "Tây Ninh"],
            "Đà Nẵng": ["Quảng Nam", "Thừa Thiên Huế"],
            # Add more mappings as needed
        }
    
    def normalize_province(self, province_name: Optional[str]) -> Optional[str]:
        """Normalize province name."""
        if not isinstance(province_name, str) or not province_name.strip():
            return None
        
        cleaned = province_name.lower().strip()
        
        # Special mappings
        if "phú quốc" in cleaned:
            return "Kiên Giang"
        if any(k in cleaned for k in ["hcm", "hồ chí minh", "ho chi minh"]):
            return "TP Hồ Chí Minh"
        if any(k in cleaned for k in ["hn", "hà nội", "ha noi"]):
            return "Hà Nội"
        if any(k in cleaned for k in ["đn", "da nang", "đà nẵng"]):
            return "Đà Nẵng"
        
        # Fuzzy match
        from difflib import SequenceMatcher
        return max(
            STANDARD_PROVINCES,
            key=lambda t: SequenceMatcher(None, cleaned, t.lower()).ratio()
        )
    
    def calculate_location_score(self, hotel: Dict[str, Any], 
                               user_places: List[str],
                               province: Optional[str] = None,
                               is_near_center: bool = False) -> float:
        """Calculate location score for a hotel."""
        # This would implement the location scoring logic
        # For now, return a placeholder score
        return 0.5


class ScoringService:
    """Service for scoring calculations."""
    
    def __init__(self):
        self.config = RecommendationConfig()
    
    def calculate_amenity_score(self, user_amenities: List[str], 
                               hotel_amenities: List[str]) -> float:
        """Calculate amenity similarity score."""
        if not user_amenities or not hotel_amenities:
            return 0.0
        
        # Simple Jaccard similarity
        user_set = set(user_amenities)
        hotel_set = set(hotel_amenities)
        intersection = user_set & hotel_set
        union = user_set | hotel_set
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_price_score(self, user_price_range: Tuple[int, int], 
                            hotel_price: int) -> float:
        """Calculate price score."""
        min_price, max_price = user_price_range
        
        if min_price <= hotel_price <= max_price:
            # Price is within range, calculate how close to center
            center = (min_price + max_price) / 2
            distance = abs(hotel_price - center)
            max_distance = (max_price - min_price) / 2
            return 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
        
        # Price is outside range, penalize
        if hotel_price < min_price:
            return 0.8  # Slightly penalize cheaper options
        else:
            return 0.2  # Heavily penalize more expensive options
    
    def calculate_rating_score(self, user_rating: float, hotel_rating: float) -> float:
        """Calculate rating similarity score."""
        if user_rating is None or hotel_rating is None:
            return 0.0
        
        diff = abs(hotel_rating - user_rating)
        if diff == 0:
            return 1.0
        elif diff <= 1:
            return 0.7
        elif diff <= 2:
            return 0.4
        else:
            return 0.1
    
    def calculate_policy_score(self, user_policies: Dict[str, Any], 
                             hotel_policies: Dict[str, Any]) -> float:
        """Calculate policy compatibility score."""
        # This would implement the policy scoring logic
        # For now, return a placeholder score
        return 0.5


class RecommendationService:
    """Main service for hotel recommendations."""
    
    def __init__(self):
        self.data_service = DataService()
        self.amenity_service = AmenityService()
        self.location_service = LocationService()
        self.scoring_service = ScoringService()
        self.config = RecommendationConfig()
    
    def recommend_hotels(self, query: HotelQuery, 
                        hotel_data: Dict[str, Any]) -> List[FinalRecommendation]:
        """Generate hotel recommendations based on query."""
        results = []
        
        for hotel_id, hotel_info in hotel_data.items():
            # Calculate individual scores
            scores = {}
            
            # Amenity score
            if query.amenities and hotel_info.get('amenities'):
                scores['amenity'] = self.scoring_service.calculate_amenity_score(
                    query.amenities, hotel_info['amenities']
                )
            
            # Location score
            if query.nearby_places or query.province:
                scores['location'] = self.location_service.calculate_location_score(
                    hotel_info, query.nearby_places, query.province, query.is_near_center
                )
            
            # Price score
            if query.price_range and hotel_info.get('price'):
                scores['price'] = self.scoring_service.calculate_price_score(
                    (query.price_range.min_price, query.price_range.max_price),
                    hotel_info['price']
                )
            
            # Rating score
            if query.stars_rating and hotel_info.get('stars_rating'):
                scores['rating'] = self.scoring_service.calculate_rating_score(
                    query.stars_rating, hotel_info['stars_rating']
                )
            
            # Policy score
            if query.policies_hotels and hotel_info.get('policies'):
                scores['policy'] = self.scoring_service.calculate_policy_score(
                    query.policies_hotels.to_dict(), hotel_info['policies']
                )
            
            # Calculate final score
            final_score = self._calculate_final_score(scores)
            
            if final_score > 0:
                recommendation = FinalRecommendation(
                    hotel_id=hotel_id,
                    final_score=final_score,
                    individual_scores=scores,
                    hotel_info=HotelInfo(**hotel_info) if hotel_info else None
                )
                results.append(recommendation)
        
        # Sort by final score and return top results
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:self.config.max_results]
    
    def _calculate_final_score(self, scores: Dict[str, float]) -> float:
        """Calculate final weighted score."""
        if not scores:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for score_type, score in scores.items():
            weight = self.config.weights.get(score_type, 0.0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


class QueryService:
    """Service for query management."""
    
    def __init__(self):
        self.queries: Dict[str, HotelQuery] = {}
    
    def create_query(self, query_data: Dict[str, Any]) -> HotelQuery:
        """Create a new hotel query."""
        return HotelQuery.from_dict(query_data)
    
    def save_query(self, query: HotelQuery) -> str:
        """Save query and return its ID."""
        self.queries[query.query_id] = query
        return query.query_id
    
    def get_query(self, query_id: str) -> Optional[HotelQuery]:
        """Get query by ID."""
        return self.queries.get(query_id)
    
    def find_queries(self, **filters) -> List[HotelQuery]:
        """Find queries matching filters."""
        results = []
        for query in self.queries.values():
            if self._matches_filters(query, filters):
                results.append(query)
        return results
    
    def _matches_filters(self, query: HotelQuery, filters: Dict[str, Any]) -> bool:
        """Check if query matches filters."""
        for key, value in filters.items():
            if not hasattr(query, key):
                continue
            
            query_value = getattr(query, key)
            
            if isinstance(query_value, list):
                if not any(item in query_value for item in (value if isinstance(value, list) else [value])):
                    return False
            elif query_value != value:
                return False
        
        return True


class AnalyticsService:
    """Service for analytics and reporting."""
    
    def __init__(self):
        self.data_service = DataService()
    
    def generate_recommendation_report(self, recommendations: List[FinalRecommendation]) -> Dict[str, Any]:
        """Generate analytics report for recommendations."""
        if not recommendations:
            return {}
        
        # Calculate statistics
        scores = [rec.final_score for rec in recommendations]
        
        # Score distribution by type
        score_types = {}
        for rec in recommendations:
            for score_type, score in rec.individual_scores.items():
                if score_type not in score_types:
                    score_types[score_type] = []
                score_types[score_type].append(score)
        
        # Calculate averages
        avg_scores = {score_type: sum(scores) / len(scores) 
                     for score_type, scores in score_types.items()}
        
        return {
            'total_recommendations': len(recommendations),
            'average_final_score': sum(scores) / len(scores),
            'min_final_score': min(scores),
            'max_final_score': max(scores),
            'average_scores_by_type': avg_scores,
            'top_hotel': recommendations[0].hotel_id if recommendations else None,
            'score_distribution': {
                'high': len([s for s in scores if s >= 0.8]),
                'medium': len([s for s in scores if 0.5 <= s < 0.8]),
                'low': len([s for s in scores if s < 0.5])
            }
        }
    
    def analyze_query_patterns(self, queries: List[HotelQuery]) -> Dict[str, Any]:
        """Analyze patterns in user queries."""
        if not queries:
            return {}
        
        # Analyze provinces
        provinces = [q.province for q in queries if q.province]
        province_counts = {}
        for province in provinces:
            province_counts[province] = province_counts.get(province, 0) + 1
        
        # Analyze price ranges
        price_ranges = [q.price_range for q in queries if q.price_range]
        avg_min_price = sum(pr.min_price for pr in price_ranges) / len(price_ranges) if price_ranges else 0
        avg_max_price = sum(pr.max_price for pr in price_ranges) / len(price_ranges) if price_ranges else 0
        
        # Analyze amenities
        all_amenities = []
        for q in queries:
            all_amenities.extend(q.amenities)
        amenity_counts = {}
        for amenity in all_amenities:
            amenity_counts[amenity] = amenity_counts.get(amenity, 0) + 1
        
        return {
            'total_queries': len(queries),
            'most_popular_provinces': sorted(province_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'average_price_range': {
                'min': avg_min_price,
                'max': avg_max_price
            },
            'most_requested_amenities': sorted(amenity_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'average_capacity': sum(q.capacity for q in queries) / len(queries)
        }

