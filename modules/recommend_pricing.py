import re
import math
from collections import defaultdict


class PriceUtils:
    """Các hàm tiện ích xử lý giá tiền."""

    @staticmethod
    def parse_price_vnd(price_str):
        """
        Chuyển đổi chuỗi giá tiền VND sang số nguyên.
        Ví dụ: "VND 1.200.000" -> 1200000
        """
        if not price_str:
            return 0
        cleaned = price_str.replace('VND', '').replace('.', '').strip()
        try:
            return int(cleaned)
        except ValueError:
            return 0


class PriceScorer:
    """Tính điểm số giá phòng khách sạn theo nhu cầu người dùng."""

    def __init__(self, user_input, beta_left=2, beta_right=10):
        self.price_range = user_input.get("price_range", (0, float("inf")))
        self.beta_left = beta_left
        self.beta_right = beta_right
        self.p = self._compute_target_price(self.price_range)

    def _compute_target_price(self, price_range):
        """Tính giá mục tiêu p từ khoảng giá (a, b)."""
        if isinstance(price_range, (list, tuple)) and len(price_range) == 2:
            a, b = price_range
            return (a + b) / 2
        elif isinstance(price_range, (int, float)):
            return price_range
        return 0

    def score_room(self, room):
        """Tính điểm giá cho một phòng."""
        hi = PriceUtils.parse_price_vnd(room.get("original_price"))
        if hi <= 0:
            return None

        di = (self.p - hi) / hi

        if self.p <= hi:
            score = math.exp(-self.beta_left * abs(di))
        else:
            score = math.exp(-self.beta_right * abs(di))

        return round(score, 4)

    def score_hotel_rooms(self, room_hotels):
        """Tính điểm giá cho toàn bộ danh sách phòng."""
        results = []
        for room in room_hotels:
            score = self.score_room(room)
            if score and score > 0:
                results.append({
                    "id": room.get("id"),
                    "room_id": room.get("room_id"),
                    "room_score_price": score
                })
        return results

    @staticmethod
    def avg_score_per_hotel(room_hotels):
        """Tính điểm giá trung bình cho từng khách sạn."""
        hotel_scores = defaultdict(list)
        for room in room_hotels:
            hotel_id = room.get("id")
            score = room.get("room_score_price", 0)
            if score and score > 0:
                hotel_scores[hotel_id].append(score)

        results = []
        for hotel_id, scores in hotel_scores.items():
            avg_score = sum(scores) / len(scores)
            results.append({
                "id": hotel_id,
                "hotel_score_price": round(avg_score, 4)
            })
        return results

    @staticmethod
    def max_score_per_hotel(room_hotels):
        """Tính điểm giá cao nhất cho từng khách sạn."""
        hotel_scores = defaultdict(list)
        for room in room_hotels:
            hotel_id = room.get("id")
            score = room.get("room_score_price", 0)
            if score and score > 0:
                hotel_scores[hotel_id].append(score)

        results = []
        for hotel_id, scores in hotel_scores.items():
            max_score = max(scores)
            results.append({
                "id": hotel_id,
                "hotel_score_price": round(max_score, 4)
            })
        return results
