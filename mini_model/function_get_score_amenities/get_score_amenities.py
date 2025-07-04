from model_classify_amenities.clustering_factilities import find_clusters
import json
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
import os
import pickle
from typing import List, Dict, Tuple, Optional
from safetensors.torch import save_file, load_file

import sys
from pathlib import Path

# Thêm thư mục gốc (mini_model) vào sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config

with open(config.get_path("merged_clusters_2.json"), "rb") as f:
    merged_clusters = json.load(f)

with open(config.get_path("feature_facilities.json"), 'r', encoding='utf-8') as f:
    feature_facilities = json.load(f)

with open(config.get_path("feature_popular_facilities.json"), 'r', encoding='utf-8') as f:
    feature_popular_facilities = json.load(f)

def convert_amenity(value):
    amenities = {
        1: "Tiện ích phòng",
        2: "Tiện ích công nghệ & kết nối",
        3: "Tiện ích phòng tắm",
        4: "Dịch vụ ẩm thực",
        5: "Tiện ích thư giãn & giải trí",
        6: "Tiện ích dành cho trẻ em",
        7: "Dịch vụ khách sạn",
        8: "Dịch vụ bảo mật & an toàn",
        9: "Dịch vụ tổ chức sự kiện",
        10: "Tiện nghi hỗ trợ đặc biệt",
        11: "Dịch vụ giải trí ngoài trời",
        12: "Dịch vụ bãi đỗ xe",
        13: "Tiện ích thiên nhiên",
        14: "Hệ thống điều hòa & sưởi ấm",
        15: "Cơ sở vật chất chung",
        16: "Dịch vụ chăm sóc sức khỏe & spa",
        17: "Dịch vụ đưa đón & phương tiện di chuyển",
        18: "Tiện ích công việc"
    }
    
    if isinstance(value, int):
        return amenities.get(value, "Không tìm thấy")
    elif isinstance(value, str):
        return next((k for k, v in amenities.items() if v == value), "Không tìm thấy")
    return "Định dạng không hợp lệ"


def filter_matching_elements(ids_A, B):
    try:
        return [item for item in B if item['id'] in ids_A]  # Lọc các phần tử B có id trong A
    except:
        return [item for item in B if item['id_room'] in ids_A]
    

def get_main_cluster(clusters, threshold=0.1):
    # Sắp xếp cụm theo giá trị giảm dần
    clusters.sort(key=lambda x: x[1], reverse=True)

    # Lấy cụm có giá trị lớn nhất
    main_cluster = [clusters[0][0]]

    # Kiểm tra nếu cụm thứ hai có giá trị gần bằng cụm đầu tiên
    if len(clusters) > 1 and abs(clusters[0][1] - clusters[1][1]) <= threshold * clusters[0][1]:
        main_cluster.append(clusters[1][0])

    return main_cluster

def get_facilities_from_cluster(main_clusters, merged_clusters):
    facilities = []
    for cluster in main_clusters:
        if cluster in merged_clusters:
            facilities.extend(merged_clusters[cluster])  # Thêm các tiện ích vào danh sách

    return list(set(facilities))  


def suggest_related_facilities(user_input):
    """
    Đề xuất các tiện ích liên quan trong cùng cụm với tiện ích người dùng nhập.
    
    Args:
        user_input (str): Tiện ích mà người dùng nhập.
    
    Returns:
        list: Danh sách các tiện ích liên quan trong cùng cụm.
    """
    # Xác định cụm của tiện ích nhập vào
    cluster = get_main_cluster(find_clusters([user_input]), threshold=0.1)
    facilities_in_cluster = get_facilities_from_cluster(cluster, merged_clusters)

    suggested_facilities = [fac for fac in facilities_in_cluster if fac != user_input]

    return cluster, suggested_facilities



def count_number_occurrences(data, selected_numbers):
    """
    Tính số lần xuất hiện của các số trong selected_numbers
    trong danh sách các tiện ích của các khách sạn

    Parameters:
    - data (dict): Dữ liệu khách sạn, với key là id và value là dict
                   chứa các key: 'id', 'name', 'address', 'value'
    - selected_numbers (list): Danh sách các số cần đếm

    Returns:
    - selected_counts (dict): Danh sách các số và số lần xuất hiện
    - total_count (int): Tổng số lần xuất hiện của các số
    """
    number_counts = defaultdict(int)
    for amenities in data["value"].values():
        if amenities is None:
            continue  # Bỏ qua nếu là None
        for num in amenities:
            number_counts[num] += 1  # Tăng số lần xuất hiện của số đó

    occurrences = dict(number_counts) 
    selected_counts = {num: occurrences.get(num, 0) for num in selected_numbers} 
    total_count = sum(selected_counts.values())  
    return selected_counts, total_count

def get_score_designated_services(facilities = None, selected_numbers = None, type = "popular"):
    """
    Tính điểm cho các tiện ích được chỉ định
    (là các tiện ích có trong danh sách facilities)
    dựa trên số lần xuất hiện của các số trong selected_numbers
    trong danh sách các tiện ích của các khách sạn

    Parameters:
    - facilities (list): Danh sách các tiện ích được chỉ định
    - selected_numbers (list): Danh sách các số cần đếm

    Returns:
    - result (list): Danh sách các kết quả, mỗi kết quả là một dictionary
                     với hai key: 'id' và 'score'
    """
    result = []
    if facilities is None:
        if type == "popular":
            facilities = feature_popular_facilities
        else:
            facilities = feature_facilities

    for facility in facilities:
        result.append({'id': facility['id'], 'score': count_number_occurrences(facility, selected_numbers)})

    result = sorted(result, key=lambda x: x['score'][1], reverse=True)
    return result

def extract_scores(result, use_tuple=False):
    """
    Trích xuất các id và điểm số từ kết quả

    Parameters:
    - result (list): Danh sách các kết quả, mỗi kết quả là một dictionary
                     với hai key: 'id' và 'score'
    - use_tuple (bool): Nếu True, lấy giá trị thứ hai từ tuple điểm số
                       (ví dụ (18, 56.61702487091125) => 56.61702487091125)
                       Nếu False, lấy tổng giá trị của dictionary điểm số
                       (ví dụ ({'4': 18}, 56.61702487091125) => 18)

    Returns:
    - ids (list): Danh sách các id
    - scores (list): Danh sách các điểm số
    """
    ids, scores = [], []
    for item in result:
        ids.append(item['id'])
        
        score = item['score']
        
        if isinstance(score, tuple):  # Trường hợp tuple ({'4': 18}, 18)
            if use_tuple:
                scores.append(score[1])  # Lấy tổng điểm từ tuple
            else:
                scores.append(sum(score[0].values()))  # Tổng điểm từ dictionary
        else:  # Trường hợp số (56.61702487091125)
            scores.append(score)
    
    return ids, scores

def normalize_scores(scores, beta=1.0, use_median=False):
    """
    Chuẩn hóa điểm số bằng hàm log-logistic để ưu ái điểm cao và phạt mạnh điểm thấp.

    Parameters:
    - scores (list): Danh sách các điểm số
    - beta (float): Hệ số điều chỉnh độ dốc và mức độ lệch phải (mặc định = 1.0)
    - use_median (bool): Nếu True thì dùng median thay vì mean làm x0

    Returns:
    - scaled_scores (list): Danh sách các điểm số đã chuẩn hóa trong [0, 1]
    """
    # Chuyển scores thành numpy array
    scores_array = np.array(scores)

    # Kiểm tra nếu scores rỗng hoặc không hợp lệ
    if len(scores_array) == 0 or np.all(np.isnan(scores_array)):
        return [0.0] * len(scores)

    # Tính x0 (mean hoặc median)
    x0 = np.median(scores_array) if use_median else np.mean(scores_array)

    # Đảm bảo x0 > 0 để tránh chia cho 0
    if x0 <= 0:
        x0 = 1e-6  # Giá trị nhỏ để tránh lỗi

    # Xử lý trường hợp scores <= 0 (gán giá trị rất nhỏ)
    scores_array = np.where(scores_array <= 0, 1e-6, scores_array)

    # Tính (x / x0)^beta
    ratio = scores_array / x0
    powered_ratio = np.power(ratio, beta)

    # Áp dụng hàm log-logistic
    scaled = powered_ratio / (1 + powered_ratio)

    # Đảm bảo kết quả trong [0, 1]
    scaled = np.clip(scaled, 0, 1)

    return scaled.tolist()

def calculate_score(result1, result2, weights=[0.5, 0.5], beta=1.0, use_median=False):
    """
    Tính điểm tổng hợp từ hai nhóm kết quả với trọng số tương ứng.

    Parameters:
    - result1: **[{'id': str, 'score': float}]**
    - result2: **[{'id': str, 'score': float}]**
    - weights: Danh sách trọng số tương ứng cho hai tập kết quả

    Returns:
    - Danh sách kết quả cuối cùng **[{'id': str, 'final_score': float}]**, sắp xếp giảm dần theo điểm
    """
    # Trích xuất dữ liệu cho từng nhóm
    ids1, scores1 = extract_scores(result1)
    ids2, scores2 = extract_scores(result2)

    # Áp dụng Min-Max Scaler cho từng tập kết quả
    norm_scores1 = normalize_scores(scores1, beta=beta, use_median=use_median)
    norm_scores2 = normalize_scores(scores2, beta=beta, use_median=use_median)

    # Tính tổng điểm có trọng số
    final_scores = {}
    all_ids = set(ids1 + ids2)

    for id_ in all_ids:
        score = 0
        if id_ in ids1:
            score += norm_scores1[ids1.index(id_)] * weights[0]
        if id_ in ids2:
            score += norm_scores2[ids2.index(id_)] * weights[1]
        final_scores[id_] = score

    # Chuyển kết quả thành danh sách và sắp xếp theo điểm giảm dần
    final_results = [{'id': k, 'final_score': v} for k, v in final_scores.items()]
    final_results.sort(key=lambda x: x['final_score'], reverse=True)

    return final_results

def get_score_services(user_input: list[int] | list[str], List_ids: list = None, weights: list = [0.5, 0.5],  beta=1.0, use_median=False, print_result = False):
    """
    Computes the scores for designated services based on user input and facilities.

    Parameters:
    - user_input (list[int] | list[str]): List of user-provided amenities as integers or strings.
    - facilities (optional): List of facilities to consider for scoring. Defaults to None.
    - weights (list): Weights for combining popular and non-popular scores. Defaults to [0.5, 0.5].

    Returns:
    - final_results (list): A list of dictionaries with 'id' and 'final_score', sorted in descending order.
    """
    if List_ids is not None:
        facilities = filter_matching_elements(List_ids, feature_facilities)
        facilities_popular = filter_matching_elements(List_ids, feature_popular_facilities)
    else:
        facilities = feature_facilities
        facilities_popular = feature_popular_facilities

    if isinstance(user_input, list) and len(user_input) > 0 and isinstance(user_input[0], str):
        selected_numbers = set(map(str, [convert_amenity(num) for num in user_input]))
    else:
        selected_numbers = user_input

    result_popular = get_score_designated_services(facilities_popular, selected_numbers = selected_numbers, type = "popular")
    result_ = get_score_designated_services(facilities, selected_numbers = selected_numbers, type = None)

    if print_result:
        print(result_popular)
        print(result_)
    final_results = calculate_score(result_popular, result_, weights, beta, use_median)

    return final_results

def calculate_hotel_scores(result_hotel, result_room, threshold=0.9, weight=[0.5, 0.5]):
    # Dictionary để lưu điểm tổng của từng khách sạn
    """
    Calculates the final scores for hotels based on the similarity scores of hotels and rooms.

    Parameters:
    - result_hotel (list): A list of dictionaries containing hotel information and similarity scores.
    - result_room (list): A list of dictionaries containing room information and similarity scores.
    - threshold (float): The minimum score for a hotel to be included in the final results.
    - weight (list): Weights for combining hotel and room scores. Defaults to [0.5, 0.5].

    Returns:
    - hotel_scores (dict): A dictionary with hotel IDs as keys and final scores as values.
    """
    hotel_scores = {}
    
    # Duyệt qua từng khách sạn trong result_hotel
    for hotel in result_hotel:
        hotel_id = hotel['hotel_id']
        hotel_score = hotel['similarity_score']
        
        # Tìm tất cả các phòng thuộc khách sạn này
        related_rooms = [
            room for room in result_room 
            if room['room_id'].startswith('RD' + hotel_id)
        ]
        
        # Trường hợp không có phòng
        if not related_rooms:
            hotel_scores[hotel_id] = hotel_score * threshold
        else:
            # Lấy điểm cao nhất của các phòng
            max_room_score = max(room['similarity_score'] for room in related_rooms)
            # Tính điểm tổng theo trọng số
            hotel_scores[hotel_id] = (hotel_score * weight[0]) + (max_room_score * weight[1])
    
    return hotel_scores


class HotelSimilarityRecommender:
    def __init__(
        self,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        use_gpu: bool = True,
        model_dir: str = "/content",
        type: str = "hotel",
        batch_size: int = 512,
        embedding_dim: Optional[int] = None,
        faiss_metric: str = 'IP',  # Inner Product
        normalize_embeddings: bool = True
    ):
        """
        Khởi tạo HotelRecommender với các tham số tùy chỉnh.

        Args:
            model_name (str): Tên mô hình Sentence-BERT.
            use_gpu (bool): Sử dụng GPU nếu có.
            model_dir (str): Thư mục lưu trữ vector, dữ liệu và chỉ mục FAISS.
            batch_size (int): Kích thước batch khi mã hóa văn bản.
            embedding_dim (int, optional): Kích thước vector embedding (tự động nếu None).
            faiss_metric (str): Loại metric cho FAISS ('IP' hoặc 'L2').
            normalize_embeddings (bool): Chuẩn hóa L2 embedding trước khi lập chỉ mục.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.faiss_metric = faiss_metric
        self.normalize_embeddings = normalize_embeddings
        self.model_dir = model_dir
        self.type = type
        self.vector_file = os.path.join(model_dir, "hotel_vectors.npy")
        self.data_file = os.path.join(model_dir, "hotel_data.pkl")
        self.index_file = os.path.join(model_dir, "faiss_index.bin")
        self.safetensors_file = os.path.join(model_dir, "model.safetensors")
        self.hotel_vectors = None
        self.hotel_data = None
        self.index = None

        # Khởi tạo hoặc tải mô hình Sentence-BERT
        if os.path.exists(self.safetensors_file):
            print(f"Loading Sentence-BERT model from {self.safetensors_file}")
            self._load_safetensors_model()
        else:
            print(f"Initializing Sentence-BERT model: {model_name}")
            self.model = SentenceTransformer(model_name)
            if use_gpu and torch.cuda.is_available():
                self.model = self.model.to('cuda')
                print(f"Using GPU for Sentence-BERT model: {model_name}")
            else:
                print(f"Using CPU for Sentence-BERT model: {model_name}")

        # Lấy kích thước embedding nếu không chỉ định
        if self.embedding_dim is None:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def _load_safetensors_model(self):
        """Tải mô hình Sentence-BERT từ file SafeTensors."""
        state_dict = load_file(self.safetensors_file)
        self.model = SentenceTransformer(self.model_name)
        self.model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            print("Moved loaded model to GPU")
        else:
            print("Loaded model on CPU")

    def preprocess_amenities(self, amenities: List[str]) -> List[str]:
        """Xử lý danh sách tiện nghi: loại bỏ khoảng trắng, chuyển thành chữ thường."""
        return [amenity.strip().lower() for amenity in amenities if amenity.strip()]

    def embed_amenities(
        self,
        amenities: List[str],
        input_amenities: Optional[List[str]] = None,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Tạo embedding cho danh sách tiện nghi, có thể sử dụng trọng số tùy chỉnh.

        Args:
            amenities (List[str]): Danh sách tiện nghi cần mã hóa.
            input_amenities (List[str], optional): Tiện nghi đầu vào để tính độ tương đồng.
            weights (np.ndarray, optional): Trọng số cho các embedding.

        Returns:
            np.ndarray: Vector embedding trung bình.
        """
        cleaned_amenities = self.preprocess_amenities(amenities)
        if not cleaned_amenities:
            return np.zeros(self.embedding_dim)

        embeddings = self.model.encode(
            cleaned_amenities,
            show_progress_bar=False,
            batch_size=self.batch_size,
            convert_to_numpy=True
        )

        if input_amenities:
            input_embedding = self.model.encode(
                input_amenities,
                batch_size=self.batch_size,
                convert_to_numpy=True
            )
            similarities = np.dot(embeddings, input_embedding.T) / (
                np.linalg.norm(embeddings, axis=1)[:, None] * np.linalg.norm(input_embedding, axis=1)
            )
            weights = np.clip(similarities.max(axis=1), 0.1, 1.0)
            weights[weights > 0.95] *= 2.0  # Nhân đôi trọng số cho khớp gần chính xác
            weights /= weights.sum()

        return np.average(embeddings, axis=0, weights=weights) if weights is not None else np.mean(embeddings, axis=0)

    def batch_embed_amenities(self, all_amenities: List[List[str]]) -> List[np.ndarray]:
        """
        Tạo embedding cho danh sách tiện nghi của nhiều khách sạn.

        Args:
            all_amenities (List[List[str]]): Danh sách tiện nghi cho mỗi khách sạn.

        Returns:
            List[np.ndarray]: Danh sách vector embedding cho mỗi khách sạn.
        """
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

    def fit(self, hotels: List[Dict[str, any]], recompute_vectors: bool = False):
        """
        Huấn luyện mô hình: tạo embedding và lập chỉ mục FAISS.

        Args:
            hotels (List[Dict[str, any]]): Danh sách dữ liệu khách sạn.
            recompute_vectors (bool): Tính lại vector thay vì load từ file.
        """
        self.hotel_data = hotels

        # Load hoặc tính vector
        if os.path.exists(self.vector_file) and not recompute_vectors:
            print(f"Loading precomputed vectors from {self.vector_file}")
            self.hotel_vectors = np.load(self.vector_file)
        else:
            print("Computing vectors for hotels...")
            all_amenities = [hotel.get('amenities', []) for hotel in hotels]
            self.hotel_vectors = self.batch_embed_amenities(all_amenities)
            np.save(self.vector_file, self.hotel_vectors)
            print(f"Saved vectors to {self.vector_file}")

        # Chuẩn bị chỉ mục FAISS
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

        """Lưu dữ liệu, vector và chỉ mục FAISS."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if self.model is not None:
            state_dict = self.model.state_dict()
            save_file(state_dict, self.safetensors_file)
            print(f"Saved Sentence-BERT model to {self.safetensors_file}")
        

        if self.hotel_vectors is not None:
            np.save(self.vector_file, self.hotel_vectors)

        if self.hotel_data is not None:
            with open(self.data_file, 'wb') as f:
                pickle.dump(self.hotel_data, f)

        if self.index is not None:
            faiss.write_index(self.index, self.index_file)

        print(f"Model saved to {self.model_dir}")

    def load_model(self):

        """Tải dữ liệu, vector và chỉ mục FAISS."""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory {self.model_dir} does not exist")
        
        if os.path.exists(self.safetensors_file):
            self._load_safetensors_model()
        else:
            print(f"SafeTensors file {self.safetensors_file} not found, initializing new model")
            self.model = SentenceTransformer(self.model_name)
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                print(f"Initialized new model {self.model_name} on GPU")
            else:
                print(f"Initialized new model {self.model_name} on CPU")

        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                self.hotel_data = pickle.load(f)
            print(f"Loaded hotel data from {self.data_file}")
        else:
            raise FileNotFoundError(f"Hotel data file {self.data_file} not found")

        if os.path.exists(self.vector_file):
            self.hotel_vectors = np.load(self.vector_file)
            print(f"Loaded vectors from {self.vector_file}")
        else:
            raise FileNotFoundError(f"Vector file {self.vector_file} not found")

        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            print(f"Loaded Faiss index from {self.index_file}")
        else:
            raise FileNotFoundError(f"Faiss index file {self.index_file} not found")

    def predict_topK(
        self,
        input_amenities: List[str],
        top_k: int = 10,
        similarity_threshold: float = 0.90,
        normalization_factor_base: float = 10.0
    ) -> List[Dict[str, any]]:
        """
        Dự đoán các khách sạn/phòng phù hợp dựa trên tiện nghi đầu vào.

        Args:
            input_amenities (List[str]): Danh sách tiện nghi đầu vào.
            type (str): Loại dự đoán ('hotel' hoặc 'room').
            top_k (int): Số lượng kết quả trả về.
            similarity_threshold (float): Ngưỡng Cosine để tính tiện nghi khớp.
            normalization_factor_base (float): Cơ số để điều chỉnh hệ số chuẩn hóa.

        Returns:
            List[Dict[str, any]]: Danh sách kết quả với thông tin khách sạn/phòng.
        """
        input_vector = self.embed_amenities(input_amenities, input_amenities).astype('float32').reshape(1, -1)
        if self.normalize_embeddings:
            faiss.normalize_L2(input_vector)
        
        similarities, indices = self.index.search(input_vector, top_k)
        results = []
        input_embedding = self.model.encode(input_amenities, batch_size=self.batch_size, convert_to_numpy=True)

        for sim, idx in zip(similarities[0], indices[0]):
            hotel = self.hotel_data[idx]
            hotel_amenities = self.preprocess_amenities(hotel.get('amenities', []))
            num_amenities = len(hotel_amenities)

            # Tính tiện nghi khớp
            hotel_embeddings = self.model.encode(hotel_amenities, batch_size=self.batch_size, convert_to_numpy=True)
            sim_matrix = np.dot(hotel_embeddings, input_embedding.T) / (
                np.linalg.norm(hotel_embeddings, axis=1)[:, None] * np.linalg.norm(input_embedding, axis=1)
            )
            matched_indices = np.where(sim_matrix.max(axis=1) > similarity_threshold)[0]
            matched_amenities = [hotel_amenities[i] for i in matched_indices]
            num_matches = len(matched_amenities)

            # Điều chỉnh hệ số chuẩn hóa
            normalization_factor = min(np.sqrt(max(1, num_amenities)) / np.sqrt(normalization_factor_base), 1.5)
            if num_matches == 0:
                adjusted_sim = sim * normalization_factor * 0.5 
            else:
                adjusted_sim = sim * normalization_factor * (num_matches / max(1, len(input_amenities)))


            result = {
                'similarity_score': float(adjusted_sim),
                'original_similarity': float(sim),
                'num_amenities': num_amenities,
                'matched_amenities': matched_amenities,
                'num_matches': num_matches,
                'amenities': hotel.get('amenities', [])
            }

            if self.type == 'hotel':
                result['hotel_id'] = hotel.get('id')
            else:
                result['room_id'] = hotel.get('id_room')

            results.append(result)

        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    def predict_assignID(
        self,
        input_amenities: List[str],
        hotel_ids: List[str],
        similarity_threshold: float = 0.90,
        normalization_factor_base: float = 10.0
    ) -> List[Dict[str, any]]:
        """
        Dự đoán độ tương đồng của các khách sạn/phòng được chỉ định dựa trên tiện nghi đầu vào.

        Args:
            input_amenities (List[str]): Danh sách tiện nghi đầu vào.
            hotel_ids (List[str]): Danh sách ID khách sạn cần dự đoán.
            type (str): Loại dự đoán ('hotel' hoặc 'room').
            similarity_threshold (float): Ngưỡng Cosine để tính tiện nghi khớp.
            normalization_factor_base (float): Cơ số để điều chỉnh hệ số chuẩn hóa.

        Returns:
            List[Dict[str, any]]: Danh sách kết quả với thông tin khách sạn/phòng.
        """
        # Tạo embedding cho tiện nghi đầu vào
        input_vector = self.embed_amenities(input_amenities, input_amenities).astype('float32').reshape(1, -1)
        if self.normalize_embeddings:
            faiss.normalize_L2(input_vector)
        
        input_embedding = self.model.encode(input_amenities, batch_size=self.batch_size, convert_to_numpy=True)
        results = []

        # Tạo ánh xạ từ hotel_id sang chỉ số trong hotel_data
        hotel_id_to_idx = {hotel.get('id' if self.type == 'hotel' else 'id_room'): idx 
                          for idx, hotel in enumerate(self.hotel_data)}

        for hotel_id in hotel_ids:
            # Kiểm tra hotel_id có trong dữ liệu không
            if hotel_id not in hotel_id_to_idx:
                #print(f"Cảnh báo: {self.type} '{hotel_id}' không có trong dữ liệu, trả về kết quả rỗng")
                result = {
                    'similarity_score': 0.0,
                    'original_similarity': 0.0,
                    'num_amenities': 0,
                    'matched_amenities': [],
                    'num_matches': 0,
                    'amenities': []
                }
                if self.type == 'hotel':
                    result['hotel_id'] = hotel_id
                else:
                    result['room_id'] = hotel_id
                results.append(result)
                continue

            idx = hotel_id_to_idx[hotel_id]
            hotel = self.hotel_data[idx]
            hotel_amenities = self.preprocess_amenities(hotel.get('amenities', []))
            num_amenities = len(hotel_amenities)

            # Tính độ tương đồng Cosine
            hotel_vector = self.hotel_vectors[idx].astype('float32').reshape(1, -1)
            if self.normalize_embeddings:
                faiss.normalize_L2(hotel_vector)
            sim = np.dot(input_vector, hotel_vector.T)[0, 0]

            # Tính tiện nghi khớp
            if num_amenities > 0:
                hotel_embeddings = self.model.encode(hotel_amenities, batch_size=self.batch_size, convert_to_numpy=True)
                sim_matrix = np.dot(hotel_embeddings, input_embedding.T) / (
                    np.linalg.norm(hotel_embeddings, axis=1)[:, None] * np.linalg.norm(input_embedding, axis=1)
                )
                matched_indices = np.where(sim_matrix.max(axis=1) > similarity_threshold)[0]
                matched_amenities = [hotel_amenities[i] for i in matched_indices]
                num_matches = len(matched_amenities)
            else:
                matched_amenities = []
                num_matches = 0

            normalization_factor = min(np.sqrt(max(1, num_amenities)) / np.sqrt(normalization_factor_base), 1.5)
            if num_matches == 0:
                adjusted_sim = sim * normalization_factor * 0.5 
            else:
                adjusted_sim = sim * normalization_factor * (num_matches / max(1, len(input_amenities)))

            result = {
                'similarity_score': float(adjusted_sim),
                'original_similarity': float(sim),
                'num_amenities': num_amenities,
                'matched_amenities': matched_amenities,
                'num_matches': num_matches,
                'amenities': hotel.get('amenities', [])
            }

            if self.type == 'hotel':
                result['hotel_id'] = hotel.get('id')
            else:
                result['room_id'] = hotel.get('id_room')

            results.append(result)

        # Sắp xếp theo similarity_score (giảm dần)
        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
        return results



    def reinitialize_model(self, model_name: str):
        """
        Tái khởi tạo với mô hình Sentence-BERT khác.

        Args:
            model_name (str): Tên mô hình Sentence-BERT mới.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            print(f"Reinitialized with model: {model_name} on GPU")
        else:
            print(f"Reinitialized with model: {model_name} on CPU")
        self.hotel_vectors = None
        self.index = None



    

    
