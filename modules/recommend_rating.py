import math
import re
import numpy as np


class StarsScorer:
    """Tính điểm dựa trên số sao khách sạn (stars_score)."""

    @staticmethod
    def score(user_stars, hotel_stars):
        if user_stars is None or hotel_stars is None:
            return 0.0

        diff = abs(hotel_stars - user_stars)
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.7
        elif diff == 2:
            return 0.4
        else:
            return 0.1


class SubRatingScorer:
    """
    Tính điểm rating_score dựa trên các tiêu chí phụ:
    staff, facilities, cleanliness, comfort, value, location, wifi.
    """

    KEY_MAP = {
        "staff": "Nhân viên phục vụ",
        "facilities": "Tiện nghi",
        "cleanliness": "Sạch sẽ",
        "comfort": "Thoải mái",
        "value": "Đáng giá tiền",
        "location": "Địa điểm",
        "wifi": "WiFi miễn phí"
    }

    @staticmethod
    def normalize_score(score, max_score=10):
        """Chuẩn hóa điểm về [0,1]."""
        if score is None:
            return 0.0
        try:
            if isinstance(score, str):
                score = score.replace(",", ".")
            return float(score) / max_score
        except:
            return 0.0

    @classmethod
    def score(cls, user_sub_rate, hotel_sub_rate):
        """
        user_sub_rate: dict user mong muốn {en_key: value}
        hotel_sub_rate: dict điểm khách sạn {vi_key: value}
        """
        weighted_sum = 0.0
        total_weight = 0.0

        # Tính tổng trọng số S_total
        for en_key, vi_key in cls.KEY_MAP.items():
            hotel_val = cls.normalize_score(hotel_sub_rate.get(vi_key))
            user_val = cls.normalize_score(user_sub_rate.get(en_key))

            if en_key in user_sub_rate and user_sub_rate[en_key] is not None:
                weighted_sum += hotel_val * 2
                total_weight += 2
            else:
                weighted_sum += hotel_val
                total_weight += 1

        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight  # chuẩn hóa về [0,1]


class RatingScorer:
    """Tổng hợp điểm rating theo công thức (3.11)–(3.13)."""

    def __init__(self, user_input):
        self.user_stars = user_input.get("stars_rating")
        self.user_rating = SubRatingScorer.normalize_score(user_input.get("rating"))
        self.user_sub_rate = user_input.get("sub_rate", {})

    def score_hotel(self, hotel):
        hotel_id = hotel.get("id")
        hotel_stars = hotel.get("stars_rating")
        hotel_info = hotel.get("info", {})

        # Điểm sao
        stars_score = StarsScorer.score(self.user_stars, hotel_stars)

        # Điểm tổng quan rating
        rating = SubRatingScorer.normalize_score(hotel_info.get("rating"))

        # Điểm sub_rating
        sub_rating = SubRatingScorer.score(self.user_sub_rate, hotel_info.get("sub_rate", {}))

        # Tổng hợp
        final_score = (stars_score + sub_rating + rating) / 3.0
        final_score = max(0.0, min(round(final_score, 4), 1.0))

        return {
            "id": hotel_id,
            "url": hotel_info.get("url"),
            "score_rating": final_score
        }

    def score_hotels(self, hotels):
        return [self.score_hotel(hotel) for hotel in hotels]
