"""
Refactored amenity recommendation module.
"""

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import faiss
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

from config import settings, AMENITIES, SERVICE_ID_TO_GROUP, SERVICE_TEXT_TO_GROUP
from schemas import RecommendationResult


class DataManager:
    """Manages data loading and configuration."""
    
    def __init__(self):
        self.merged_clusters = None
        self.feature_facilities = None
        self.feature_popular_facilities = None
        self._load_data()
    
    def _load_data(self):
        """Load required data files."""
        try:
            merged_clusters_path = settings.get_path("merged_clusters_2.json")
            with open(merged_clusters_path, "rb") as f:
                self.merged_clusters = json.load(f)
            
            feature_facilities_path = settings.get_path("feature_facilities.json")
            with open(feature_facilities_path, 'r', encoding='utf-8') as f:
                self.feature_facilities = json.load(f)
            
            feature_popular_facilities_path = settings.get_path("feature_popular_facilities.json")
            with open(feature_popular_facilities_path, 'r', encoding='utf-8') as f:
                self.feature_popular_facilities = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load data files: {e}")


class AmenityConverter:
    """Handles amenity conversion between IDs and names."""
    
    @classmethod
    def convert_amenity(cls, value: Union[int, str]) -> Union[str, int]:
        """Convert between amenity ID and name."""
        if isinstance(value, int):
            return AMENITIES.get(value, "Không tìm thấy")
        elif isinstance(value, str):
            for amenity_id, amenity_name in AMENITIES.items():
                if amenity_name == value:
                    return amenity_id
            return "Không tìm thấy"
        return "Định dạng không hợp lệ"


class ClusterManager:
    """Manages clustering operations for facilities."""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
    
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
            #     cluster, self.data_manager.merged_clusters
            # )
            # suggested_facilities = [fac for fac in facilities_in_cluster if fac != user_input]
            # return cluster, suggested_facilities
            return [], []
        except Exception as e:
            print(f"Error in suggest_related_facilities: {e}")
            return [], []


class ScoreCalculator:
    """Handles scoring calculations for facilities and hotels."""
    
    @staticmethod
    def filter_matching_elements(ids_a: List[int], items_b: List[Dict]) -> List[Dict]:
        """Filter elements in B that have IDs present in A."""
        if not items_b:
            return []
        
        id_field = 'id' if 'id' in items_b[0] else 'id_room'
        return [item for item in items_b if item.get(id_field) in ids_a]
    
    @staticmethod
    def count_number_occurrences(data: Dict, selected_numbers: List) -> Tuple[Dict, int]:
        """Count occurrences of selected numbers in facility data."""
        number_counts = defaultdict(int)
        
        for amenities in data["value"].values():
            if amenities is None:
                continue
            for num in amenities:
                number_counts[num] += 1
        
        occurrences = dict(number_counts)
        selected_counts = {num: occurrences.get(num, 0) for num in selected_numbers}
        total_count = sum(selected_counts.values())
        
        return selected_counts, total_count
    
    def get_score_designated_services(self, facilities: Optional[List] = None,
                                    selected_numbers: Optional[List] = None,
                                    facility_type: str = "popular") -> List[Dict]:
        """Calculate scores for designated services."""
        if facilities is None:
            data_manager = DataManager()
            facilities = (data_manager.feature_popular_facilities 
                         if facility_type == "popular" 
                         else data_manager.feature_facilities)
        
        if selected_numbers is None:
            selected_numbers = []
        
        result = []
        for facility in facilities:
            score = self.count_number_occurrences(facility, selected_numbers)
            result.append({'id': facility['id'], 'score': score})
        
        return sorted(result, key=lambda x: x['score'][1], reverse=True)
    
    @staticmethod
    def extract_scores(result: List[Dict], use_tuple: bool = False) -> Tuple[List, List]:
        """Extract IDs and scores from result list."""
        ids, scores = [], []
        
        for item in result:
            ids.append(item['id'])
            score = item['score']
            
            if isinstance(score, tuple):
                if use_tuple:
                    scores.append(score[1])
                else:
                    scores.append(sum(score[0].values()))
            else:
                scores.append(score)
        
        return ids, scores
    
    @staticmethod
    def normalize_scores(scores: List[float], beta: float = 1.0, 
                        use_median: bool = False) -> List[float]:
        """Normalize scores using log-logistic function."""
        scores_array = np.array(scores)
        
        if len(scores_array) == 0 or np.all(np.isnan(scores_array)):
            return [0.0] * len(scores)
        
        x0 = np.median(scores_array) if use_median else np.mean(scores_array)
        if x0 <= 0:
            x0 = 1e-6
        
        scores_array = np.where(scores_array <= 0, 1e-6, scores_array)
        ratio = scores_array / x0
        powered_ratio = np.power(ratio, beta)
        scaled = powered_ratio / (1 + powered_ratio)
        
        return np.clip(scaled, 0, 1).tolist()
    
    def calculate_score(self, result1: List[Dict], result2: List[Dict],
                       weights: List[float] = [0.5, 0.5], beta: float = 1.0,
                       use_median: bool = False) -> List[Dict]:
        """Calculate combined scores from two result sets."""
        ids1, scores1 = self.extract_scores(result1)
        ids2, scores2 = self.extract_scores(result2)
        
        norm_scores1 = self.normalize_scores(scores1, beta=beta, use_median=use_median)
        norm_scores2 = self.normalize_scores(scores2, beta=beta, use_median=use_median)
        
        final_scores = {}
        all_ids = set(ids1 + ids2)
        
        for id_ in all_ids:
            score = 0
            if id_ in ids1:
                score += norm_scores1[ids1.index(id_)] * weights[0]
            if id_ in ids2:
                score += norm_scores2[ids2.index(id_)] * weights[1]
            final_scores[id_] = score
        
        final_results = [{'id': k, 'final_score': v} for k, v in final_scores.items()]
        return sorted(final_results, key=lambda x: x['final_score'], reverse=True)


class HotelScoreService:
    """Service for calculating hotel scores based on amenities."""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.score_calculator = ScoreCalculator()
        self.amenity_converter = AmenityConverter()
    
    def get_score_services(self, user_input: Union[List[int], List[str]],
                          list_ids: Optional[List] = None,
                          weights: List[float] = [0.5, 0.5],
                          beta: float = 1.0, use_median: bool = False,
                          print_result: bool = False) -> List[Dict]:
        """Compute scores for services based on user input."""
        if list_ids is not None:
            facilities = self.score_calculator.filter_matching_elements(
                list_ids, self.data_manager.feature_facilities
            )
            facilities_popular = self.score_calculator.filter_matching_elements(
                list_ids, self.data_manager.feature_popular_facilities
            )
        else:
            facilities = self.data_manager.feature_facilities
            facilities_popular = self.data_manager.feature_popular_facilities
        
        # Convert string inputs to amenity IDs if necessary
        if (isinstance(user_input, list) and len(user_input) > 0 and 
            isinstance(user_input[0], str)):
            selected_numbers = set(
                str(self.amenity_converter.convert_amenity(num)) for num in user_input
            )
        else:
            selected_numbers = user_input
        
        result_popular = self.score_calculator.get_score_designated_services(
            facilities_popular, selected_numbers=selected_numbers, facility_type="popular"
        )
        result_general = self.score_calculator.get_score_designated_services(
            facilities, selected_numbers=selected_numbers, facility_type="general"
        )
        
        if print_result:
            print("Popular results:", result_popular)
            print("General results:", result_general)
        
        return self.score_calculator.calculate_score(
            result_popular, result_general, weights, beta, use_median
        )
    
    @staticmethod
    def calculate_hotel_scores(result_hotel: List[Dict], result_room: List[Dict],
                             threshold: float = 0.9, 
                             weight: List[float] = [0.5, 0.5]) -> Dict[str, float]:
        """Calculate final hotel scores combining hotel and room scores."""
        hotel_scores = {}
        
        for hotel in result_hotel:
            hotel_id = hotel['hotel_id']
            hotel_score = hotel['similarity_score']
            
            related_rooms = [
                room for room in result_room
                if room['room_id'].startswith('RD' + hotel_id)
            ]
            
            if not related_rooms:
                hotel_scores[hotel_id] = hotel_score * threshold
            else:
                max_room_score = max(room['similarity_score'] for room in related_rooms)
                hotel_scores[hotel_id] = (
                    hotel_score * weight[0] + max_room_score * weight[1]
                )
        
        return hotel_scores


class HotelSimilarityRecommender:
    """Advanced hotel recommendation system using semantic similarity."""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 use_gpu: bool = True, model_dir: str = "./models",
                 recommendation_type: str = "hotel", batch_size: int = 512,
                 embedding_dim: Optional[int] = None, faiss_metric: str = 'IP',
                 normalize_embeddings: bool = True):
        """Initialize the hotel recommender."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.faiss_metric = faiss_metric
        self.normalize_embeddings = normalize_embeddings
        self.model_dir = model_dir
        self.recommendation_type = recommendation_type
        
        # File paths
        self.vector_file = os.path.join(model_dir, "hotel_vectors.npy")
        self.data_file = os.path.join(model_dir, "hotel_data.pkl")
        self.index_file = os.path.join(model_dir, "faiss_index.bin")
        self.safetensors_file = os.path.join(model_dir, "model.safetensors")
        
        # Initialize attributes
        self.hotel_vectors = None
        self.hotel_data = None
        self.index = None
        self.model = None
        
        self._initialize_model(use_gpu)
    
    def _initialize_model(self, use_gpu: bool):
        """Initialize or load the Sentence-BERT model."""
        if os.path.exists(self.safetensors_file):
            print(f"Loading model from {self.safetensors_file}")
            self._load_safetensors_model()
        else:
            print(f"Initializing new model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            if use_gpu and torch.cuda.is_available():
                self.model = self.model.to('cuda')
                print("Using GPU for model")
            else:
                print("Using CPU for model")
        
        if self.embedding_dim is None:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def _load_safetensors_model(self):
        """Load model from SafeTensors file."""
        try:
            state_dict = load_file(self.safetensors_file)
            self.model = SentenceTransformer(self.model_name)
            self.model.load_state_dict(state_dict)
            
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                print("Loaded model on GPU")
            else:
                print("Loaded model on CPU")
        except Exception as e:
            print(f"Error loading SafeTensors model: {e}")
            self._initialize_new_model()
    
    def _initialize_new_model(self):
        """Initialize a new model when loading fails."""
        self.model = SentenceTransformer(self.model_name)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
    
    @staticmethod
    def preprocess_amenities(amenities: List[str]) -> List[str]:
        """Preprocess amenities: strip whitespace and convert to lowercase."""
        return [amenity.strip().lower() for amenity in amenities if amenity.strip()]
    
    def embed_amenities(self, amenities: List[str],
                       input_amenities: Optional[List[str]] = None,
                       weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Create embeddings for amenities with optional custom weights."""
        cleaned_amenities = self.preprocess_amenities(amenities)
        if not cleaned_amenities:
            return np.zeros(self.embedding_dim)
        
        embeddings = self.model.encode(
            cleaned_amenities,
            show_progress_bar=False,
            batch_size=self.batch_size,
            convert_to_numpy=True
        )
        
        if input_amenities and weights is None:
            input_embedding = self.model.encode(
                input_amenities,
                batch_size=self.batch_size,
                convert_to_numpy=True
            )
            
            similarities = np.dot(embeddings, input_embedding.T) / (
                np.linalg.norm(embeddings, axis=1)[:, None] * 
                np.linalg.norm(input_embedding, axis=1)
            )
            
            weights = np.clip(similarities.max(axis=1), 0.1, 1.0)
            weights[weights > 0.95] *= 2.0  # Double weight for near-exact matches
            weights /= weights.sum()
        
        if weights is not None:
            return np.average(embeddings, axis=0, weights=weights)
        else:
            return np.mean(embeddings, axis=0)
    
    def batch_embed_amenities(self, all_amenities: List[List[str]]) -> List[np.ndarray]:
        """Create embeddings for multiple hotels' amenities efficiently."""
        flat_amenities = []
        hotel_indices = []
        
        for i, amenities in enumerate(all_amenities):
            cleaned_amenities = self.preprocess_amenities(amenities)
            if cleaned_amenities:
                flat_amenities.extend(cleaned_amenities)
                hotel_indices.extend([i] * len(cleaned_amenities))
        
        if not flat_amenities:
            return [np.zeros(self.embedding_dim) for _ in all_amenities]
        
        embeddings = self.model.encode(
            flat_amenities,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        hotel_vectors = []
        for i in range(len(all_amenities)):
            hotel_embs = embeddings[np.array(hotel_indices) == i]
            if len(hotel_embs) == 0:
                hotel_vectors.append(np.zeros(self.embedding_dim))
            else:
                hotel_vectors.append(np.mean(hotel_embs, axis=0))
        
        return hotel_vectors
    
    def fit(self, hotels: List[Dict[str, Any]], recompute_vectors: bool = False):
        """Train the model by creating embeddings and FAISS index."""
        self.hotel_data = hotels
        
        # Load or compute vectors
        if os.path.exists(self.vector_file) and not recompute_vectors:
            print(f"Loading precomputed vectors from {self.vector_file}")
            self.hotel_vectors = np.load(self.vector_file)
        else:
            print("Computing vectors for hotels...")
            all_amenities = [hotel.get('amenities', []) for hotel in hotels]
            self.hotel_vectors = self.batch_embed_amenities(all_amenities)
            np.save(self.vector_file, self.hotel_vectors)
            print(f"Saved vectors to {self.vector_file}")
        
        # Set up FAISS index
        self.hotel_vectors = np.array(self.hotel_vectors).astype('float32')
        
        if self.faiss_metric == 'IP':
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.faiss_metric == 'L2':
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unsupported FAISS metric: {self.faiss_metric}")
        
        if self.normalize_embeddings:
            faiss.normalize_L2(self.hotel_vectors)
        
        self.index.add(self.hotel_vectors)
    
    def save_model(self):
        """Save model, data, vectors, and FAISS index."""
        os.makedirs(self.model_dir, exist_ok=True)
        
        if self.model is not None:
            state_dict = self.model.state_dict()
            save_file(state_dict, self.safetensors_file)
            print(f"Saved model to {self.safetensors_file}")
        
        if self.hotel_vectors is not None:
            np.save(self.vector_file, self.hotel_vectors)
        
        if self.hotel_data is not None:
            with open(self.data_file, 'wb') as f:
                pickle.dump(self.hotel_data, f)
        
        if self.index is not None:
            faiss.write_index(self.index, self.index_file)
        
        print(f"Model saved to {self.model_dir}")
    
    def load_model(self):
        """Load model, data, vectors, and FAISS index."""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory {self.model_dir} does not exist")
        
        # Load model
        if os.path.exists(self.safetensors_file):
            self._load_safetensors_model()
        else:
            print("SafeTensors file not found, initializing new model")
            self._initialize_new_model()
        
        # Load data files
        required_files = [
            (self.data_file, 'hotel_data'),
            (self.vector_file, 'hotel_vectors'),
            (self.index_file, 'index')
        ]
        
        for file_path, attr_name in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{attr_name.title()} file {file_path} not found")
        
        with open(self.data_file, 'rb') as f:
            self.hotel_data = pickle.load(f)
        
        self.hotel_vectors = np.load(self.vector_file)
        self.index = faiss.read_index(self.index_file)
        
        print(f"Loaded model from {self.model_dir}")
    
    def predict_top_k(self, input_amenities: List[str], top_k: int = 10,
                     similarity_threshold: float = 0.90,
                     normalization_factor_base: float = 10.0) -> List[RecommendationResult]:
        """Predict top K similar hotels/rooms based on input amenities."""
        input_vector = self.embed_amenities(
            input_amenities, input_amenities
        ).astype('float32').reshape(1, -1)
        
        if self.normalize_embeddings:
            faiss.normalize_L2(input_vector)
        
        similarities, indices = self.index.search(input_vector, top_k)
        results = []
        
        for sim, idx in zip(similarities[0], indices[0]):
            hotel = self.hotel_data[idx]
            hotel_amenities = hotel.get('amenities', [])
            
            # Calculate similarity metrics
            metrics = self._calculate_similarity_metrics(
                input_amenities, hotel_amenities, similarity_threshold
            )
            
            # Adjust similarity score
            adjusted_sim = self._adjust_similarity_score(
                sim, metrics, len(input_amenities), normalization_factor_base
            )
            
            result = RecommendationResult(
                hotel_id=hotel.get('id'),
                similarity_score=float(adjusted_sim),
                original_similarity=float(sim),
                amenities=hotel_amenities,
                num_amenities=metrics['num_amenities'],
                matched_amenities=metrics['matched_amenities'],
                num_matches=metrics['num_matches']
            )
            
            results.append(result)
        
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)[:top_k]
    
    def _calculate_similarity_metrics(self, input_amenities: List[str],
                                    hotel_amenities: List[str],
                                    similarity_threshold: float) -> Dict[str, Any]:
        """Calculate similarity metrics between input and hotel amenities."""
        hotel_amenities_clean = self.preprocess_amenities(hotel_amenities)
        num_amenities = len(hotel_amenities_clean)
        
        if num_amenities == 0:
            return {
                'num_amenities': 0,
                'matched_amenities': [],
                'num_matches': 0
            }
        
        # Calculate matched amenities
        input_embedding = self.model.encode(
            input_amenities, batch_size=self.batch_size, convert_to_numpy=True
        )
        hotel_embeddings = self.model.encode(
            hotel_amenities_clean, batch_size=self.batch_size, convert_to_numpy=True
        )
        
        sim_matrix = np.dot(hotel_embeddings, input_embedding.T) / (
            np.linalg.norm(hotel_embeddings, axis=1)[:, None] * 
            np.linalg.norm(input_embedding, axis=1)
        )
        
        matched_indices = np.where(sim_matrix.max(axis=1) > similarity_threshold)[0]
        matched_amenities = [hotel_amenities_clean[i] for i in matched_indices]
        
        return {
            'num_amenities': num_amenities,
            'matched_amenities': matched_amenities,
            'num_matches': len(matched_amenities)
        }
    
    def _adjust_similarity_score(self, similarity: float, metrics: Dict[str, Any],
                                num_input_amenities: int,
                                normalization_factor_base: float) -> float:
        """Adjust similarity score based on various factors."""
        num_amenities = metrics['num_amenities']
        num_matches = metrics['num_matches']
        
        normalization_factor = min(
            np.sqrt(max(1, num_amenities)) / np.sqrt(normalization_factor_base), 1.5
        )
        
        if num_matches == 0:
            return similarity * normalization_factor * 0.5
        else:
            match_ratio = num_matches / max(1, num_input_amenities)
            return similarity * normalization_factor * match_ratio


# Legacy function wrappers for backward compatibility
def convert_amenity(value: Union[int, str]) -> Union[str, int]:
    """Legacy wrapper for backward compatibility."""
    return AmenityConverter.convert_amenity(value)


def filter_matching_elements(ids_A: List[int], B: List[Dict]) -> List[Dict]:
    """Legacy wrapper for backward compatibility."""
    return ScoreCalculator.filter_matching_elements(ids_A, B)


def get_main_cluster(clusters: List[Tuple], threshold: float = 0.1) -> List:
    """Legacy wrapper for backward compatibility."""
    data_manager = DataManager()
    cluster_manager = ClusterManager(data_manager)
    return cluster_manager.get_main_cluster(clusters, threshold)


def get_facilities_from_cluster(main_clusters: List, merged_clusters: Dict) -> List[str]:
    """Legacy wrapper for backward compatibility."""
    data_manager = DataManager()
    cluster_manager = ClusterManager(data_manager)
    return cluster_manager.get_facilities_from_cluster(main_clusters, merged_clusters)


def suggest_related_facilities(user_input: str) -> Tuple[List, List[str]]:
    """Legacy wrapper for backward compatibility."""
    data_manager = DataManager()
    cluster_manager = ClusterManager(data_manager)
    return cluster_manager.suggest_related_facilities(user_input)


def count_number_occurrences(data: Dict, selected_numbers: List) -> Tuple[Dict, int]:
    """Legacy wrapper for backward compatibility."""
    return ScoreCalculator.count_number_occurrences(data, selected_numbers)


def get_score_designated_services(facilities: Optional[List] = None,
                                selected_numbers: Optional[List] = None,
                                type: str = "popular") -> List[Dict]:
    """Legacy wrapper for backward compatibility."""
    score_calculator = ScoreCalculator()
    return score_calculator.get_score_designated_services(
        facilities, selected_numbers, type
    )


def extract_scores(result: List[Dict], use_tuple: bool = False) -> Tuple[List, List]:
    """Legacy wrapper for backward compatibility."""
    return ScoreCalculator.extract_scores(result, use_tuple)


def normalize_scores(scores: List[float], beta: float = 1.0, 
                    use_median: bool = False) -> List[float]:
    """Legacy wrapper for backward compatibility."""
    return ScoreCalculator.normalize_scores(scores, beta, use_median)


def calculate_score(result1: List[Dict], result2: List[Dict],
                   weights: List[float] = [0.5, 0.5], beta: float = 1.0,
                   use_median: bool = False) -> List[Dict]:
    """Legacy wrapper for backward compatibility."""
    score_calculator = ScoreCalculator()
    return score_calculator.calculate_score(result1, result2, weights, beta, use_median)


def get_score_services(user_input: Union[List[int], List[str]],
                      List_ids: Optional[List] = None,
                      weights: List[float] = [0.5, 0.5],
                      beta: float = 1.0, use_median: bool = False,
                      print_result: bool = False) -> List[Dict]:
    """Legacy wrapper for backward compatibility."""
    service = HotelScoreService()
    return service.get_score_services(
        user_input, List_ids, weights, beta, use_median, print_result
    )


def calculate_hotel_scores(result_hotel: List[Dict], result_room: List[Dict],
                          threshold: float = 0.9, 
                          weight: List[float] = [0.5, 0.5]) -> Dict[str, float]:
    """Legacy wrapper for backward compatibility."""
    return HotelScoreService.calculate_hotel_scores(
        result_hotel, result_room, threshold, weight
    )

