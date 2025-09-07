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




class RoomProcessing:
    """Các hàm xử lý liên quan đến phòng khách sạn."""

    room_type_list = [
        'Bungalow', 'Single Room', 'Suite', 'Family Room', 'Group Room',
        'Small Family Room', 'Double Room', 'Deluxe', 'Apartment/Studio/Penthouse',
        'Villa / Bungalow', 'Mansion/Residence', 'Family/Group Room',
        'Other', 'Chalet', 'Villa', 'Twin Room ', 'Cabin', 'Dormitory'
    ]

    room_level_list = [
        'Standard', 'Junior', 'Royal', 'Premium', 'Premier', 
        'Luxury', 'Executive', 'Signature', 'Senior', 'VIP'
    ]

    bed_type_list = [
        "giường đôi cực lớn", "giường đôi lớn", "giường đôi",
        "giường đơn", "giường sofa", "giường tầng", "giường nệm futon kiểu Nhật"
    ]

    @staticmethod
    def extract_bed_counts(text: str) -> dict:
        """Trích xuất số lượng giường từ mô tả text."""
        if not isinstance(text, str):
            return {}

        bed_types = sorted(RoomProcessing.bed_type_list, key=len, reverse=True)
        bed_count = defaultdict(int)

        for bed in bed_types:
            pattern = rf"(\d+)\s+{re.escape(bed)}"
            matches = re.findall(pattern, text)
            total = sum(int(m) for m in matches)
            if total:
                bed_count[bed] += total
                text = re.sub(pattern, "", text)

        return dict(bed_count)

    @staticmethod
    def normalize_area(input_str):
        """Chuẩn hoá diện tích về m²."""
        if not isinstance(input_str, str):
            return None
        input_str = input_str.strip().lower().replace(" ", "").replace(",", ".")
        match = re.match(r"([0-9]*\.?[0-9]+)(m2|m²|km2|km²)?", input_str)
        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2)
        if unit in ['km2', 'km²']:
            return value * 1_000_000
        return value

    @staticmethod
    def find_best_match(input_str, choices):
        """Tìm best match trong danh sách."""
        match = process.extractOne(input_str, choices)
        return match[0] if match else None


class RoomScorer:
    """Tính điểm số phù hợp của phòng với nhu cầu người dùng."""

    def __init__(self, user_input):
        self.user_capacity = user_input.get('capacity')
        self.user_room_type = RoomProcessing.find_best_match(user_input.get('room_type', ''), RoomProcessing.room_type_list)
        self.user_room_level = RoomProcessing.find_best_match(user_input.get('room_level', ''), RoomProcessing.room_level_list)
        self.user_bed = RoomProcessing.extract_bed_counts(user_input.get('bed_type', ''))
        self.user_area = RoomProcessing.normalize_area(user_input.get('area', ''))
        self.user_view = user_input.get('room_view', [])
        self.user_amenities = user_input.get('amenities', [])

    def score_room(self, room):
        """Tính điểm số cho một phòng."""
        score = 0
        max_score = 0

        # === Loại phòng / sức chứa ===
        if self.user_capacity or self.user_room_type:
            max_score += 2
            if (room.get('room_type') == self.user_room_type or
                room.get('cap_room_type') == self.user_room_type or
                (self.user_capacity and self.user_capacity == room.get('capacity'))):
                score += 2

        # === Hạng phòng ===
        if self.user_room_level:
            max_score += 1
            if room.get('room_level') == self.user_room_level:
                score += 1

        # === Loại giường ===
        if self.user_bed:
            max_score += 1
            bed_types = room.get('bed_types', {})
            for bed in self.user_bed.keys():
                if bed in bed_types:
                    score += 1
                    break

        # === Diện tích ===
        if self.user_area and room.get('room_area(m²)'):
            max_score += 1
            if room['room_area(m²)'] - 5 <= self.user_area <= room['room_area(m²)'] + 5:
                score += 1

        # === View phòng ===
        if self.user_view:
            max_score += 1
            room_views = room.get('room_view', [])
            if not isinstance(room_views, list):
                room_views = [room_views]

            missing_views = len([v for v in self.user_view if v not in room_views])
            s_view = 1 - missing_views / len(self.user_view)
            score += s_view

        # === Tiện nghi phòng ===
        if self.user_amenities:
            max_score += 1
            room_amenities = room.get('amenities', [])
            missing_amenities = len([a for a in self.user_amenities if a not in room_amenities])
            s_amen = 1 - missing_amenities / len(self.user_amenities)
            score += s_amen

        # Chuẩn hoá về [0,1]
        return round(score / max_score, 2) if max_score > 0 else 0.0

    def score_hotel_rooms(self, hotel_rooms):
        """Tính điểm số cho tất cả phòng của khách sạn."""
        results = []
        for room in hotel_rooms:
            results.append({
                'id': room.get('id'),
                'room_id': room.get('room_id'),
                'room_score': self.score_room(room)
            })
        return results
