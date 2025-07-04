import numpy as np
import json

import sys
from pathlib import Path

# Thêm thư mục gốc (mini_model) vào sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config


try:
    with open(config.get_path("feature_sub_rate.json"), 'r', encoding='utf-8') as f:
        feature_sub_rate = json.load(f)
except FileNotFoundError:
    print("File feature_sub_rate.json not found. Please check the file path.")
    feature_sub_rate = []

try:
    with open(config.get_path("weights_criteria_utilities.json"), 'r', encoding='utf-8') as f:
        weights_criteria_utilities = json.load(f)
except FileNotFoundError:
    print("File weights_criteria_utilities.json not found. Please check the file path.")
    weights_criteria_utilities = {}



def get_dynamic_weights(selected_utilities, utility_scores=weights_criteria_utilities):
    """
    Tạo trọng số động dựa trên các tiện ích mà người dùng chọn.

    Parameters:
    - selected_utilities (list of str): Danh sách các tiện ích người dùng quan tâm.
    - utility_scores (dict): Bảng trọng số giữa tiện ích và hạng mục đánh giá (mặc định lấy từ file JSON).

    Returns:
    - dict: Trọng số trung bình của các hạng mục đánh giá liên quan đến tiện ích đã chọn.
    """
    category_weights = {}
    
    for utility in selected_utilities:
        if utility in utility_scores:
            for category, weight in utility_scores[utility].items():
                if category in category_weights:
                    category_weights[category].append(weight)
                else:
                    category_weights[category] = [weight]
    
    return {category: np.mean(weights) for category, weights in category_weights.items()}


def normalize_hotel_scores(hotel_data):
    """
    Chuẩn hóa điểm đánh giá của từng khách sạn về khoảng [0, 1].

    Parameters:
    - hotel_data (dict): Dữ liệu điểm đánh giá từng hạng mục của một khách sạn.

    Returns:
    - dict: Dữ liệu đã được chuẩn hóa về khoảng 0-1 cho các trường số.
    """
    normalized_data = {}
    for key, value in hotel_data.items():
        try:
            normalized_data[key] = float(value.replace(",", ".")) / 10
        except (ValueError, AttributeError):
            normalized_data[key] = value / 10
    return normalized_data


def compute_hotel_score(hotel_data, selected_utilities):
    """
    Tính điểm tổng hợp cho khách sạn dựa trên trọng số động từ tiện ích được chọn.

    Parameters:
    - hotel_data (dict): Dữ liệu thô về điểm số từng tiêu chí của khách sạn.
    - selected_utilities (list of str): Danh sách tiện ích người dùng quan tâm.

    Returns:
    - float: Điểm tổng hợp của khách sạn.
    """
    category_weights = get_dynamic_weights(selected_utilities)
    #normalized_data = normalize_hotel_scores(hotel_data)
    
    total_weight = sum(category_weights.values())
    final_score = 0
    for k in hotel_data.keys():
        if k in category_weights.keys():
            temp = float(hotel_data[k].replace(",", ".")) / 10 if isinstance(hotel_data[k], str) else hotel_data[k] / 10
            if temp is None or temp == 0:
                continue
            final_score += temp * (category_weights[k])
    return final_score / total_weight


def get_score_sub_rate(query: list[str], feature_sub_rate=feature_sub_rate):
    """
    Tính điểm và xếp hạng khách sạn dựa trên truy vấn tiện ích người dùng.

    Parameters:
    - query (list of str): Danh sách tiện ích người dùng muốn ưu tiên.
    - feature_sub_rate (list of dict): Danh sách thông tin khách sạn kèm theo điểm các hạng mục.

    Returns:
    - list of dict: Danh sách khách sạn gồm 'id' và 'score', được sắp xếp giảm dần theo điểm.
    """
    result = []
    for data in feature_sub_rate:
        result.append({'id': data['id'], 'score': compute_hotel_score(data, query)})
    
    return sorted(result, key=lambda x: x['score'], reverse=True)
