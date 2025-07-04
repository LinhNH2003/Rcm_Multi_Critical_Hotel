import os
import json
import pandas as pd
import numpy as np
import unicodedata
from difflib import SequenceMatcher
import re
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from collections import defaultdict
from rapidfuzz import process


import sys
from pathlib import Path

# Thêm thư mục gốc (mini_model) vào sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config



with open(config.get_path('hotel_data_room.json'), 'r', encoding='utf-8') as f:
    feature_detail_room = json.load(f)


### **get_room_info_score**
room_type_list = ['Bungalow', 'Single Room', 'Suite',
 'Family Room', 'Group Room', 'Small Family Room',
 'Double Room', 'Deluxe', 'Apartment/Studio/Penthouse',
 'Villa / Bungalow', 'Mansion/Residence', 'Family/Group Room',
 'Other', 'Chalet', 'Villa', 'Twin Room ', 'Cabin', 'Dormitory']

room_level_list = ['Standard', 'Junior', 'Royal', 'Premium', 'Premier', 'Luxury', 'Executive', 'Signature', 'Senior', 'VIP']
bed_type_list = [
        "giường đôi cực lớn",
        "giường đôi lớn",
        "giường đôi",
        "giường đơn",
        "giường sofa",
        "giường tầng",
        "giường nệm futon kiểu Nhật"
    ]

views_list = [
    "Ban công",
    "Hoàn toàn nằm ở tầng trệt",
    "Hướng nhìn sân trong",
    "Khu vực tiếp khách",
    "Khu vực phòng ăn",
    "Khu vực ăn uống ngoài trời",
    "Nhìn ra biển",
    "Nhìn ra hồ",
    "Nhìn ra hồ bơi",
    "Nhìn ra núi",
    "Nhìn ra sông",
    "Nhìn ra thành phố",
    "Nhìn ra vườn",
    "Nhìn ra địa danh nổi tiếng",
    "Sân hiên",
    "Sân trong",
    "Tầm nhìn ra khung cảnh"
]



def classify_resort_room(cap, bed_types=None):
    if bed_types:
        if "giường đơn" in bed_types and bed_types.get("giường đơn", 0) == 2:
            return "Twin Room"
        if any("giường đôi" in bed.lower() for bed in bed_types):
            return "Double Room"
    if np.isnan(cap):
        return None
    elif cap == 1:
        return 'Single Room'
    elif cap == 2:
        return ['Double Room', 'Twin Room']
    elif 3 <= cap <= 4:
        return 'Small Family Room'
    elif 5 <= cap <= 6:
        return 'Family Room'
    elif 7 <= cap <= 10:
        return 'Group Room'
    elif 11 <= cap <= 20:
        return 'Villa / Bungalow'
    else:
        return 'Mansion/Residence'
    


def extract_bed_counts(text: str) -> dict:
    if not isinstance(text, str):
        return {}
    
    # Ưu tiên chuỗi dài hơn trước
    bed_types = sorted([
        "giường đôi cực lớn",
        "giường đôi lớn",
        "giường đôi",
        "giường đơn",
        "giường sofa",
        "giường tầng",
        "giường nệm futon kiểu Nhật"
    ], key=len, reverse=True)

    bed_count = defaultdict(int)

    for bed in bed_types:
        pattern = rf"(\d+)\s+{re.escape(bed)}"
        matches = re.findall(pattern, text)
        total = sum(int(m) for m in matches)
        if total:
            bed_count[bed] += total
            # Loại bỏ đoạn đã bắt được để tránh trùng lặp với chuỗi ngắn hơn
            text = re.sub(pattern, "", text)

    return dict(bed_count)


def normalize_area(input_str):
    """
    Normalize a given area input string into a numeric value.

    Parameters
    ----------
    input_str : str
        The input string to be normalized.

    Returns
    -------
    float
        The normalized value of the given area in square meters.
    """
    if not isinstance(input_str, str):
        return None

    input_str = input_str.strip().lower().replace(" ", "").replace(",", ".")
    
    # Tìm số thực (có thể có phần thập phân)
    match = re.match(r"([0-9]*\.?[0-9]+)(m2|m²|km2|km²)?", input_str)
    
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)

    if unit in ['km2', 'km²']:
        return value * 1_000_000  # km² → m²
    else:
        return value  # đã là m²




def find_best_match(input_str, choices):
    match = process.extractOne(input_str, choices)
    return match[0] if match else None



def get_room_info_score(user_input, room_hotels):
    list_recommend_room_types = set()

    user_room_capacity = user_input.get('capacity')
    user_room_type = find_best_match(user_input.get('room_type', ''), room_type_list)
    user_room_level = find_best_match(user_input.get('room_level', ''), room_level_list)
    user_bed = extract_bed_counts(user_input.get('bed_type', ''))
    user_area = normalize_area(user_input.get('area', ''))
    user_view = user_input.get('room_view', [])

    if user_room_capacity:
        list_recommend_room_types.add(user_room_capacity)
    if user_room_type:
        list_recommend_room_types.update(user_room_type if isinstance(user_room_type, list) else [user_room_type])
    if user_room_level:
        list_recommend_room_types.update(user_room_level if isinstance(user_room_level, list) else [user_room_level])

    # Tính max_score dựa trên user_input
    max_score = 0
    if user_room_capacity or user_room_type:
        max_score += 2
    if user_room_level:
        max_score += 1
    if user_bed:
        max_score += 1
    if user_area and any(room.get('room_area(m²)') for room in room_hotels):
        max_score += 1
    if user_view:
        if isinstance(user_view, list):
            max_score += len(user_view)
        else:
            max_score += 1

    result = []

    for room in room_hotels:
        room_score = 0

        if (room.get('room_type') in list_recommend_room_types or
            room.get('cap_room_type') in list_recommend_room_types or
            (user_room_capacity and user_room_capacity == room.get('capacity'))):
            room_score += 2

        if user_room_level and room.get('room_level') == user_room_level:
            room_score += 1

        bed_types = room.get('bed_types', {})
        if user_bed and user_bed in list(bed_types.keys()):
            room_score += 1

        room_area = room.get('room_area(m²)')
        if user_area and room_area:
            if room_area - 5 <= user_area <= room_area + 5:
                room_score += 1

        room_views = room.get('room_view', [])
        if isinstance(user_view, list):
            for view in user_view:
                if view in room_views:
                    room_score += 1
        elif isinstance(user_view, str) and user_view in room_views:
            room_score += 1

        score = round(room_score / max_score, 2) if max_score > 0 else 0.0

        result.append({
            'id': room.get('id'),
            'room_id': room.get('room_id'),
            'room_score': score
        })

    return result



