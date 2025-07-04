from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
import sys
import os
import json
from collections import defaultdict


def convert_group(input_value, category: str = "services" or "criteria", model: str = "shorten" or "expand"):    
    """
    Convert between service or criteria ID and text, or vice versa, or expand a group to a list of IDs or texts.

    Parameters
    ----------
    input_value : int or str
        The value to convert. If int, it is the ID of the service or criteria. If str, it is the name of the service or criteria.
    category : str, optional
        The category of the input value. Can be "services" or "criteria". Default is "services".
    model : str, optional
        The model to use for the conversion. Can be "shorten" or "expand". Default is "shorten".
    
    Returns
    -------
    int, str, or list
        The converted value. If model is "shorten", it is the ID of the service or criteria. If model is "expand", it is a list of IDs or texts of the service or criteria.
    """
    if model == "shorten":
        # Dịch vụ gốc và ánh xạ sang nhóm
        services_mapping = {
            1: "Tiện ích phòng nghỉ", 3: "Tiện ích phòng nghỉ", 14: "Tiện ích phòng nghỉ",
            2: "Tiện ích công nghệ & làm việc", 18: "Tiện ích công nghệ & làm việc",
            4: "Dịch vụ ẩm thực", 13: "Tiện ích thiên nhiên",
            5: "Tiện ích giải trí & spa", 11: "Tiện ích giải trí & spa", 16: "Tiện ích giải trí & spa",
            6: "Tiện ích dành cho trẻ em",
            7: "Dịch vụ & cơ sở vật chất chung", 15: "Dịch vụ & cơ sở vật chất chung",
            8: "Dịch vụ bảo mật & an toàn",
            9: "Dịch vụ sự kiện & hỗ trợ đặc biệt", 10: "Dịch vụ sự kiện & hỗ trợ đặc biệt",
            12: "Dịch vụ vận chuyển & bãi đỗ xe", 17: "Dịch vụ vận chuyển & bãi đỗ xe"
        }
        
        services_text_mapping = {
            "Tiện ích phòng": "Tiện ích phòng nghỉ",
            "Tiện ích công nghệ & kết nối": "Tiện ích công nghệ & làm việc",
            "Tiện ích phòng tắm": "Tiện ích phòng nghỉ",
            "Dịch vụ ẩm thực": "Dịch vụ ẩm thực",
            "Tiện ích thư giãn & giải trí": "Tiện ích giải trí & spa",
            "Tiện ích dành cho trẻ em": "Tiện ích dành cho trẻ em",
            "Dịch vụ khách sạn": "Dịch vụ & cơ sở vật chất chung",
            "Dịch vụ bảo mật & an toàn": "Dịch vụ bảo mật & an toàn",
            "Dịch vụ tổ chức sự kiện": "Dịch vụ sự kiện & hỗ trợ đặc biệt",
            "Tiện nghi hỗ trợ đặc biệt": "Dịch vụ sự kiện & hỗ trợ đặc biệt",
            "Dịch vụ giải trí ngoài trời": "Tiện ích giải trí & spa",
            "Dịch vụ bãi đỗ xe": "Dịch vụ vận chuyển & bãi đỗ xe",
            "Tiện ích thiên nhiên": "Tiện ích giải trí & spa",
            "Hệ thống điều hòa & sưởi ấm": "Tiện ích phòng nghỉ",
            "Cơ sở vật chất chung": "Dịch vụ & cơ sở vật chất chung",
            "Dịch vụ chăm sóc sức khỏe & spa": "Tiện ích giải trí & spa",
            "Dịch vụ đưa đón & phương tiện di chuyển": "Dịch vụ vận chuyển & bãi đỗ xe",
            "Tiện ích công việc": "Tiện ích công nghệ & làm việc"
        }
        
        # Tiêu chí gốc và ánh xạ sang nhóm
        criteria_mapping = {
            1: "Vị trí & môi trường", 14: "Vị trí & môi trường", 15: "Vị trí & môi trường",
            2: "Phòng nghỉ & tiện nghi", 4: "Phòng nghỉ & tiện nghi", 9: "Phòng nghỉ & tiện nghi", 13: "Phòng nghỉ & tiện nghi",
            3: "Sạch sẽ",
            5: "Dịch vụ & nhân viên", 6: "Dịch vụ & nhân viên",
            7: "Wi-Fi",
            8: "Ẩm thực",
            10: "Cơ sở vật chất",
            11: "Giá trị",
            12: "An ninh",
            16: "Cơ sở vật chất"  # parking tích hợp vào facilities
        }
        # "location", "room", "cleanliness", "comfort", "service", "staff", "wifi", "food", "bathroom", "parking", "value", "facilities", "air conditioning", "view", "environment", "security"
        criteria_text_mapping = {
            "location": "Vị trí & môi trường",
            "room": "Phòng nghỉ & tiện nghi",
            "cleanliness": "Sạch sẽ",
            "comfort": "Phòng nghỉ & tiện nghi",
            "service": "Dịch vụ & nhân viên",
            "staff": "Dịch vụ & nhân viên",
            "wifi": "Wi-Fi",
            "food": "Ẩm thực",
            "bathroom": "Phòng nghỉ & tiện nghi",
            "parking": "Cơ sở vật chất",
            "value": "Giá trị",
            "facilities": "Cơ sở vật chất",
            "air conditioning": "Phòng nghỉ & tiện nghi",
            "view": "Vị trí & môi trường",
            "environment": "Vị trí & môi trường",
            "security": "An ninh"
        }
        
        if category == "services":
            if isinstance(input_value, int):
                return services_mapping.get(input_value)
            elif isinstance(input_value, str):
                return services_text_mapping.get(input_value)
        elif category == "criteria":
            if isinstance(input_value, int):
                return criteria_mapping.get(input_value)
            elif isinstance(input_value, str):
                return criteria_text_mapping.get(input_value)
        return None
    
    elif model == "expand":
        # Dịch vụ: Ánh xạ nhóm sang số thứ tự và text
        services_group_to_indices = {
            "Tiện ích phòng nghỉ": [1, 3, 14],
            "Tiện ích công nghệ & làm việc": [2, 18],
            "Dịch vụ ẩm thực": [4],
            "Tiện ích giải trí & spa": [5, 11, 16],
            "Tiện ích dành cho trẻ em": [6],
            "Dịch vụ & cơ sở vật chất chung": [7, 15],
            "Dịch vụ bảo mật & an toàn": [8],
            "Dịch vụ sự kiện & hỗ trợ đặc biệt": [9, 10],
            "Dịch vụ vận chuyển & bãi đỗ xe": [12, 17]
        }
        
        services_group_to_text = {
            "Tiện ích phòng nghỉ": ["Tiện ích phòng", "Tiện ích phòng tắm", "Hệ thống điều hòa & sưởi ấm"],
            "Tiện ích công nghệ & làm việc": ["Tiện ích công nghệ & kết nối", "Tiện ích công việc"],
            "Dịch vụ ẩm thực": ["Dịch vụ ẩm thực"],
            "Tiện ích giải trí & spa": ["Tiện ích thư giãn & giải trí", "Dịch vụ giải trí ngoài trời", "Dịch vụ chăm sóc sức khỏe & spa"],
            "Tiện ích dành cho trẻ em": ["Tiện ích dành cho trẻ em"],
            "Dịch vụ & cơ sở vật chất chung": ["Dịch vụ khách sạn", "Cơ sở vật chất chung"],
            "Dịch vụ bảo mật & an toàn": ["Dịch vụ bảo mật & an toàn"],
            "Dịch vụ sự kiện & hỗ trợ đặc biệt": ["Dịch vụ tổ chức sự kiện", "Tiện nghi hỗ trợ đặc biệt"],
            "Dịch vụ vận chuyển & bãi đỗ xe": ["Dịch vụ bãi đỗ xe", "Dịch vụ đưa đón & phương tiện di chuyển"]
        }
        
        # "Vị trí & môi trường", "Phòng nghỉ & tiện nghi", "Sạch sẽ", "Dịch vụ & nhân viên", "Wi-Fi", "Ẩm thực", "Cơ sở vật chất" , "An ninh", "Giá trị"
        # Tiêu chí: Ánh xạ nhóm sang số thứ tự và text
        criteria_group_to_indices = {
            "Vị trí & môi trường": [1, 14, 15],
            "Phòng nghỉ & tiện nghi": [2, 4, 9, 13],
            "Sạch sẽ": [3],
            "Dịch vụ & nhân viên": [5, 6],
            "Wi-Fi": [7],
            "Ẩm thực": [8],
            "Cơ sở vật chất": [10, 16],
            "An ninh": [12],
            "Giá trị": [11]
        }
        
        criteria_group_to_text = {
            "Vị trí & môi trường": ["location", "view", "environment"],
            "Phòng nghỉ & tiện nghi": ["room", "comfort", "bathroom", "air conditioning"],
            "Sạch sẽ": ["cleanliness"],
            "Dịch vụ & nhân viên": ["service", "staff"],
            "Wi-Fi": ["wifi"],
            "Ẩm thực": ["food"],
            "Cơ sở vật chất": ["facilities", "parking"],
            "An ninh": ["security"],
            "Giá trị": ["value"]
        }
        
        if category == "services":
            return services_group_to_text.get(input_value, [])
        elif category == "criteria":
            return criteria_group_to_text.get(input_value, [])
        return []
    
def filter_matching_elements(ids_A, B):
    """
    Lọc các phần tử B có id trong A.
    Nếu trong B không có key 'id' thì sẽ lọc theo key 'id_room'.
    """
    try:
        return [item for item in B if item['id'] in ids_A]  # Lọc các phần tử B có id trong A
    except:
        return [item for item in B if item['id_room'] in ids_A]


def convert_amenity(value):
    """
    Converts an integer or string representing an amenity to its corresponding
    name or ID.

    If given an integer, returns the name of the amenity. If given a string,
    returns the ID of the amenity. If the input is not found in the amenities
    list or the input type is invalid, returns an appropriate message.

    Args:
        value (int or str): The amenity ID or name.

    Returns:
        str or int: The corresponding amenity name if input is int, ID if input
        is str. Returns "Không tìm thấy" if the amenity is not found or 
        "Định dạng không hợp lệ" if the input type is invalid.
    """

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


    
def compute_total_score(*sources, weights=None):
    # Validate inputs
    """
    Compute the total score for each id by combining scores from multiple sources with weights.

    Args:
        *sources (dict): Multiple dictionaries containing scores for each id.
        weights (list): List of weights for each source. If not provided, each source is given equal weight.

    Returns:
        list: A list of tuples containing the id and total score, sorted in descending order of score.
    """
    
    if not sources:
        return []
    if weights is None:
        weights = [1.0 / len(sources)] * len(sources)
    if len(weights) != len(sources):
        raise ValueError("Number of weights must match number of sources")
    if not all(isinstance(source, dict) for source in sources):
        raise ValueError("All sources must be dictionaries")
    
    def normalize_to_range(values_dict, min_target=0.25, max_target=1.0):
        values = list(values_dict.values())
        if not values:
            return {k: min_target for k in values_dict}
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return {k: min_target for k in values_dict}  # Avoid division by zero
        # Linearly map [min_v, max_v] to [min_target, max_target]
        return {
            k: min_target + (max_target - min_target) * (v - min_v) / (max_v - min_v)
            for k, v in values_dict.items()
        }

    # Combine scores by id with weights
    normalized_sources = [normalize_to_range(source) for source in sources]

    # Combine scores by id with weights
    total_scores = defaultdict(float)
    all_ids = set()
    for source in normalized_sources:
        all_ids.update(source.keys())
        
    for id_ in all_ids:
        if len(sources) == 2:
            # Count how many sources contain this id
            sources_with_id = sum(1 for source in sources if id_ in source)
            if sources_with_id == 1:
                # If id appears in only one source, take its score and multiply by 0.8
                for source in sources:
                    if id_ in source:
                        total_scores[id_] = source[id_] * 0.8
                        break
            else:
                # If id appears in both sources, compute weighted sum
                total_scores[id_] = sum(
                    source.get(id_, 0.0) * weight
                    for source, weight in zip(sources, weights)
                )
        else:
            # For any number of sources other than 2, compute weighted sum
            total_scores[id_] = sum(
                source.get(id_, 0.0) * weight
                for source, weight in zip(sources, weights)
            )

    # Sort by score in descending order
    return sorted(total_scores.items(), key=lambda x: x[1], reverse=True)


  

def compute_intersection_score(*sources, weights=None):
    """
    Compute the total score for each id by combining scores from multiple sources with weights.
    Only consider ids that appear in all sources.

    Args:
        *sources (dict): Multiple dictionaries containing scores for each id.
        weights (list): List of weights for each source. If not provided, each source is given equal weight.

    Returns:
        list: A list of tuples containing the id and total score, sorted in descending order of score.
    """
    if not sources:
        return []
    if weights is None:
        weights = [1.0 / len(sources)] * len(sources)
    if len(weights) != len(sources):
        raise ValueError("Number of weights must match number of sources")
    if not all(isinstance(source, dict) for source in sources):
        raise ValueError("All sources must be dictionaries")

    def normalize_to_range(values_dict, min_target=0.25, max_target=1.0):
        values = list(values_dict.values())
        if not values:
            return {k: min_target for k in values_dict}
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return {k: min_target for k in values_dict}
        return {
            k: min_target + (max_target - min_target) * (v - min_v) / (max_v - min_v)
            for k, v in values_dict.items()
        }

    normalized_sources = [normalize_to_range(source) for source in sources]

    # ⬇️ LẤY GIAO CÁC ID
    all_ids = set(normalized_sources[0].keys())
    for source in normalized_sources[1:]:
        all_ids &= set(source.keys())

    total_scores = {}
    for id_ in all_ids:
        total_scores[id_] = sum(
            source.get(id_, 0.0) * weight
            for source, weight in zip(normalized_sources, weights)
        )

    return sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
