from pathlib import Path
import json
import pickle
import pandas as pd

# Base dir mặc định
DEFAULT_BASE_DIR = Path("D:/graduate_dissertation/final")

# Relative paths mapping (chỉ lưu phần sau base)
relative_paths = {
    "feature_sub_rate.json": ["dataset", "info_hotel", "feature_sub_rate.json"],
    "weights_criteria_utilities.json": ["dataset", "sub_model", "weights_criteria_utilities.json"],
    "feature_popular_facilities.json": ["dataset", "info_hotel", "feature_popular_facilities.json"],
    "feature_facilities.json": ["dataset", "info_hotel", "feature_facilities.json"],
    "feature_url.json": ["dataset", "info_hotel", "feature_url.json"],
    "feature_allRoom.json": ["dataset", "info_hotel", "feature_allRoom.json"],
    "review_processing.json": ["mini_model", "recommenr_review_data", "review_processing.json"],
    "feature_location.json": ["dataset", "info_hotel", "feature_location.json"],
    "feature_policies.json": ["dataset", "info_hotel", "feature_policies.json"], 
    "merged_clusters_2.json": ["dataset", "sub_model", "merged_clusters_2.json"],
    "vietnamese-stopwords-dash.txt": ["dataset", "sub_model", "vietnamese-stopwords-dash.txt"],
    "model_clustering_fac.pkl": ["mini_model", "function_get_score_amenities", "model_classify_amenities", "model_clustering_fac.pkl"],
    "neighbor_province.csv": ["dataset", "sub_model", "neighbor_province.csv"],
    "vietnam_location.xlsx": ["dataset", "sub_model", "vietnam_location.xlsx"], 
    "review_quality.json": ["dataset", "review_hotel", "review_quality.json"],
    "model_facilities_hotel": ["mini_model", "function_get_score_amenities", "model_classify_amenities", "hotel"],
    "model_facilities_room": ["mini_model", "function_get_score_amenities", "model_classify_amenities", "room"],
    "hotel_data_room.json": ["dataset", "info_hotel", "hotel_data_room.json"],
    "feature_star_rating.json": ["dataset", "info_hotel", "feature_star_rating.json"],
}
def get_path(name: str, base_dir: Path = DEFAULT_BASE_DIR) -> Path:
    """
    Trả về đường dẫn tuyệt đối cho một file/tên file.

    Args:
        name (str): Tên file (chỉ lưu phần sau base).
        base_dir (Path, optional): Thư mục gốc (mặc định: DEFAULT_BASE_DIR).

    Returns:
        Path: Đường dẫn tuyệt đối đến file có tên `name`.

    Raises:
        ValueError: Nếu tên file không tồn tại.
    """
    if name not in relative_paths:
        raise ValueError(f"Path alias '{name}' not found.")
    return base_dir.joinpath(*relative_paths[name])