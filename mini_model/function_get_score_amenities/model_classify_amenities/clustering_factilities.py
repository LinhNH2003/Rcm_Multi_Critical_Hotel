import pickle
from pyvi import ViTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os


import sys
from pathlib import Path

# Thêm thư mục gốc (mini_model) vào sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config

def load_stopwords(filepath=config.get_path("vietnamese-stopwords-dash.txt")):
    return set(open(filepath, encoding="utf-8").read().split("\n"))

# --- Tải model ---
filename = config.get_path("model_clustering_fac.pkl")
def load_model(filename=filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
# --- Hàm xử lý token ---
def get_token(text, vietnamese_stopwords):
    text_tokens = ViTokenizer.tokenize(text).split()
    filtered_text = [word for word in text_tokens if word not in vietnamese_stopwords]
    return filtered_text

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

# --- Tìm cụm phù hợp cho tiện ích ---
def find_best_clusters(user_input, bm25, vectorizer, tfidf_matrix, cluster_texts, vietnamese_stopwords):
    # Tiền xử lý user input
    processed_user_input = [" ".join(get_token(query, vietnamese_stopwords)) for query in user_input]

    # BM25 Scores
    bm25_scores = {cluster: 0 for cluster in cluster_texts.keys()}
    for query in processed_user_input:
        scores = bm25.get_scores(query.split())
        for i, cluster in enumerate(cluster_texts.keys()):
            bm25_scores[cluster] += scores[i]

    # Cosine Similarity Scores
    cosine_scores = {cluster: 0 for cluster in cluster_texts.keys()}
    for query in processed_user_input:
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix)[0]

        for i, cluster in enumerate(cluster_texts.keys()):
            cosine_scores[cluster] += scores[i]

    # Kết hợp điểm BM25 + Cosine Similarity
    final_scores = {cluster: bm25_scores[cluster] + cosine_scores[cluster] for cluster in cluster_texts.keys()}

    # Sắp xếp cụm theo độ tương đồng
    sorted_clusters = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_clusters
    
# Tải model đã lưu
bm25, vectorizer, tfidf_matrix, cluster_texts = load_model()

vietnamese_stopwords = load_stopwords()

def find_clusters(user_input: list, threshold=0.1): # user_input: list of queries 
    """Tìm các cụm tiện ích phù hợp cho các query của người dùng.

    Parameters
    ----------
    user_input : list
        Danh sách các query của người dùng.

    Returns
    -------
    sorted_clusters : list of tuple
        Danh sách các cụm tiện ích được sắp xếp theo độ tương đồng.
    """
    sorted_clusters = find_best_clusters(user_input, bm25, vectorizer, tfidf_matrix, cluster_texts, vietnamese_stopwords)
    # Sắp xếp cụm theo giá trị giảm dần
    sorted_clusters.sort(key=lambda x: x[1], reverse=True)

    # Lấy cụm có giá trị lớn nhất
    main_cluster = [sorted_clusters[0][0]]

    # Kiểm tra nếu cụm thứ hai có giá trị gần bằng cụm đầu tiên
    if len(sorted_clusters) > 1 and abs(sorted_clusters[0][1] - sorted_clusters[1][1]) <= threshold * sorted_clusters[0][1]:
        main_cluster.append(sorted_clusters[1][0])

    result = [convert_amenity(int(cluster)) for cluster in main_cluster]
    return result

     
