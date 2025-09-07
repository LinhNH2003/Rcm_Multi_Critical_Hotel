import re
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Processing:
    """Các hàm xử lý / trích xuất dữ liệu text và policy."""

    # --- Regex pattern ---
    patterns = {
        "thanh_toan_truoc": r"\nThanh toán trước[^\n]*",
        "khong_can_thanh_toan_truoc": r"Không cần thanh toán trước - thanh toán tại chỗ nghỉ",
        "huy_mien_phi": r"(Hủy miễn phí trước \d{1,2} tháng \d{1,2}, \d{4})|(Miễn phí hủy đến.*? \d{1,2} tháng \d{1,2}, \d{4})",
        "khong_can_the_tin_dung": r"Không cần thẻ tín dụng",
        "linh_dong_doi_ngay": r"Linh động đổi ngày khi kế hoạch thay đổi",
        "khong_hoan_tien": r"Không hoàn tiền",
        "bao_gom_bua_sang": r"(Bao bữa sáng|Bao gồm bữa sáng|Bữa sáng [^\n]*|Giá bao gồm bữa sáng)"
    }

    # --- Room service ---
    @staticmethod
    def extract_room_service_features(text: str) -> dict:
        extracted = {}
        for key, pattern in Processing.patterns.items():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            extracted[key] = bool(match)
        return extracted

    @staticmethod
    def extract_features_string(text: str) -> dict:
        extracted = {}
        for key, pattern in Processing.patterns.items():
            match = re.search(pattern, text)
            if match:
                extracted[key] = match.group(0)
        return extracted

    # --- Age ---
    @staticmethod
    def extract_age_range(text: str):
        if pd.isna(text) or not isinstance(text, str):
            return (None, None)
        text = text.lower()

        match_range = re.search(r'từ (\d{1,3})\s*-\s*(\d{1,3}) tuổi', text)
        if match_range:
            return (int(match_range.group(1)), int(match_range.group(2)))

        match_from_to = re.search(r'từ (\d{1,3}) tuổi.*?đến (\d{1,3}) tuổi', text)
        if match_from_to:
            return (int(match_from_to.group(1)), int(match_from_to.group(2)))

        match_from = re.search(r'từ (\d{1,3}) tuổi', text)
        if match_from:
            return (int(match_from.group(1)), None)

        match_to = re.search(r'đến (\d{1,3}) tuổi', text)
        if match_to:
            return (None, int(match_to.group(1)))

        match_min = re.search(r'(độ tuổi tối thiểu|tối thiểu|độ tuổi thấp nhất).*?(\d{1,3})', text)
        if match_min:
            return (int(match_min.group(2)), None)

        return (None, None)

    @staticmethod
    def extract_child_policy_info(text: str):
        if not isinstance(text, str):
            return None

        age_pattern = r'\b(\d{1,3})\s*-\s*(\d{1,3})\s*tuổi\b|\b(\d{1,3})\s*tuổi trở lên\b|\b(\d{1,3})\s*tuổi\b'
        age_match = re.findall(age_pattern, text)
        allowed_age = (0, 18)
        for match in age_match:
            if match[0] and match[1]:
                allowed_age = (int(match[0]), int(match[1]))
            elif match[2]:
                allowed_age = (int(match[2]), 18)
            elif match[3]:
                allowed_age = (int(match[3]), 18)
        return allowed_age

    # --- Time ---
    @staticmethod
    def extract_time_range(text: str):
        if not isinstance(text, str):
            return None
        text_lower = text.lower().strip()

        if "24 giờ" in text_lower or "24/24" in text_lower or "cả ngày" in text_lower or "không có giờ giới hạn" in text_lower:
            return ("00:00", "24:00")

        dash_pattern = r'\b(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\b'
        dash_match = re.search(dash_pattern, text)
        if dash_match:
            try:
                t1 = datetime.strptime(dash_match.group(1).zfill(5), "%H:%M").strftime("%H:%M")
                t2 = datetime.strptime(dash_match.group(2).zfill(5), "%H:%M").strftime("%H:%M")
                return (t1, t2)
            except ValueError:
                return None

        time_pattern = r'\b(\d{1,2}:\d{2})\b'
        times = re.findall(time_pattern, text)
        try:
            time_objs = [datetime.strptime(t.zfill(5), "%H:%M") for t in times]
            times_std = [t.strftime("%H:%M") for t in time_objs]
        except ValueError:
            return None

        if "từ" in text_lower and "đến" in text_lower and len(times_std) >= 2:
            return (times_std[0], times_std[1])
        elif "từ" in text_lower and len(times_std) >= 1:
            start = min(times_std)
            end = max(times_std) if len(times_std) > 1 else "24:00"
            return (start, end)
        elif "đến" in text_lower and len(times_std) >= 1:
            return ("00:00", times_std[0])
        elif len(times_std) == 1:
            return (times_std[0], "24:00")
        return None

    @staticmethod
    def time_to_minutes(t: str) -> int:
        h, m = map(int, t.split(':'))
        return h * 60 + m

    @staticmethod
    def is_time_range_valid(time_range_to_check, valid_range) -> bool:
        start_check = Processing.time_to_minutes(time_range_to_check[0])
        end_check = Processing.time_to_minutes(time_range_to_check[1])
        start_valid = Processing.time_to_minutes(valid_range[0])
        end_valid = Processing.time_to_minutes(valid_range[1])

        if end_valid < start_valid:
            end_valid += 1440
        if end_check < start_check:
            end_check += 1440
        return start_valid <= start_check and end_check <= end_valid

    # --- Deposit ---
    @staticmethod
    def extract_deposit_amount(text: str):
        match = re.search(r'VND\s*([\d\.]+)', text)
        if match:
            return int(match.group(1).replace('.', ''))
        return None

    # --- Pets ---
    @staticmethod
    def check_pet_policy(policy_text: str):
        if not isinstance(policy_text, str):
            return [False, True]

        text = policy_text.lower()
        allow_pet = bool(
            re.search(r"(vật nuôi được phép|cho phép mang|có thể mang)", text)
            and not re.search(r"không\s+(cho phép|được phép).*vật nuôi", text)
        )
        free_fee = bool(
            (re.search(r"không\s+tính thêm phí", text) or
             "miễn phí" in text or "không phụ thu" in text)
            and "không miễn phí" not in text
        )
        return [allow_pet, free_fee]

    # --- TF-IDF ---
    @staticmethod
    def compute_tfidf_similarity(text1: str, text2: str) -> float:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]


class PolicyScorer:
    """Tính toán điểm số policies giữa yêu cầu của người dùng và chính sách khách sạn."""

    @staticmethod
    def compute_tfidf_similarity(text1, text2):
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    @staticmethod
    def find_similar_hotel_policies(user_input, hotel):
        """
        Tính toán điểm số policies của khách sạn và người dùng

        Args:
            user_input (dict): Yêu cầu của người dùng
            hotel (dict): Thông tin của khách sạn

        Returns:
            float: Điểm số policies của khách sạn và người dùng, nằm trong khoảng [0, 1]
        """
        hotel_policies = hotel

        # 1. Check-in/out
        check_time_in = check_time_out = True
        hotel_checkin = hotel_policies.get('Nhận phòng')
        user_checkin = Processing.extract_time_range(user_input.get('Nhận phòng'))
        if hotel_checkin and user_checkin:
            check_time_in = Processing.is_time_range_valid(user_checkin, hotel_checkin)

        hotel_checkout = hotel_policies.get('Trả phòng')
        user_checkout = Processing.extract_time_range(user_input.get('Trả phòng'))
        if hotel_checkout and user_checkout:
            check_time_out = Processing.is_time_range_valid(user_checkout, hotel_checkout)

        # 2. Giờ giới nghiêm
        check_limit_time = True
        if user_input.get('Giờ giới nghiêm') and hotel_policies.get('Giờ giới nghiêm'):
            user_limit_time = Processing.extract_time_range(user_input['Giờ giới nghiêm'])
            hotel_limit_time = hotel_policies['Giờ giới nghiêm']
            if user_limit_time and hotel_limit_time:
                check_limit_time = Processing.is_time_range_valid(user_limit_time, hotel_limit_time)

        # 3. Giờ yên lặng
        check_silent_time = True
        if user_input.get('Thời gian yên lặng') and hotel_policies.get('Thời gian yên lặng'):
            user_silent_time = Processing.extract_time_range(user_input['Thời gian yên lặng'])
            hotel_silent_time = hotel_policies['Thời gian yên lặng']
            if user_silent_time and hotel_silent_time:
                check_silent_time = Processing.is_time_range_valid(user_silent_time, hotel_silent_time)

        # 4. Độ tuổi {'hotel_age_range': hotel_age_range, 'hotel_child_age': hotel_child_age}
        check_age = True
        if hotel_policies.get('Giới hạn độ tuổi'):
            hotel_age_range = hotel_policies['Giới hạn độ tuổi']['hotel_age_range']
            user_child_age = Processing.extract_child_policy_info(user_input.get('Trẻ em và giường', '0 tuổi'))
            hotel_child_age = hotel_policies['Giới hạn độ tuổi']['hotel_child_age']
            if user_child_age and hotel_child_age:
                # So sánh xem độ tuổi người dùng yêu cầu có nằm trong khoảng khách sạn cho phép không
                check_age = (
                    user_child_age[0] >= hotel_child_age[0] and
                    user_child_age[1] <= hotel_child_age[1]
                )
            else:
                # Nếu không có thông tin rõ ràng, mặc định là phù hợp
                check_age = True

        # 5. Đặt cọc
        check_deposit = True
        if user_input.get('Đặt cọc đề phòng hư hại có thể hoàn lại') and hotel_policies.get('Đặt cọc đề phòng hư hại có thể hoàn lại'):
            user_deposit = Processing.extract_deposit_amount(user_input['Đặt cọc đề phòng hư hại có thể hoàn lại'])
            hotel_deposit = hotel_policies['Đặt cọc đề phòng hư hại có thể hoàn lại']
            if hotel_deposit and user_deposit and hotel_deposit > user_deposit:
                check_deposit = False

        # 6. Tiệc tùng
        check_party = True
        if user_input.get('Tiệc tùng'):
            if hotel_policies.get('Tiệc tùng') and re.search('Không', hotel_policies['Tiệc tùng']):
                check_party = False

        # 7. Vật nuôi
        check_pet = check_pet_miscellenous = True
        if user_input.get('Vật nuôi') and hotel_policies.get('Vật nuôi'):
            user_pet = Processing.check_pet_policy(user_input['Vật nuôi'])
            hotel_pet = hotel_policies['Vật nuôi']
            if user_pet[0] and not hotel_pet[0]:
                check_pet = False
            if user_pet[0] and not hotel_pet[1]:
                check_pet_miscellenous = False

        # 8. Nhóm
        check_group = True
        if user_input.get('Nhóm') and not hotel_policies.get('Nhóm'):
            check_group = False

        # 9. Hút thuốc
        check_smoking = True
        if user_input.get('Hút thuốc'):
            if hotel_policies.get('Hút thuốc') and re.search('Không', hotel_policies['Hút thuốc']):
                if not re.search('Không', user_input['Hút thuốc']):
                    check_smoking = False

        # 10. Room service
        (
            check_thanh_toan_truoc,
            check_khong_can_thanh_toan_truoc,
            check_huy_mien_phi,
            check_khong_can_the_tin_dung,
            check_linh_dong_doi_ngay,
            check_khong_hoan_tien,
            check_bao_gom_bua_sang
        ) = [False]*7

        if user_input.get('room_service_include') and hotel_policies.get('room_service_included'):
            user_flags = Processing.extract_room_service_features(user_input['room_service_include'])
            hotel_flags = hotel_policies['room_service_included']
            (
                check_thanh_toan_truoc,
                check_khong_can_thanh_toan_truoc,
                check_huy_mien_phi,
                check_khong_can_the_tin_dung,
                check_linh_dong_doi_ngay,
                check_khong_hoan_tien,
                check_bao_gom_bua_sang
            ) = [u == h for u, h in zip(user_flags, hotel_flags)]

        # 11. TF-IDF similarity
        shared_keys = set(user_input.keys()).intersection(set(hotel_policies.keys()))
        total_similarity = 0
        for key in shared_keys:
            user_val = user_input.get(key)
            hotel_val = hotel_policies.get(key)
            if isinstance(user_val, str) and isinstance(hotel_val, str):
                sim = PolicyScorer.compute_tfidf_similarity(user_val, hotel_val)
                total_similarity += sim

        strict_checks = [
            check_time_in, check_time_out, check_age, check_deposit,
            check_limit_time, check_silent_time, check_party, check_pet,
            check_pet_miscellenous, check_group, check_smoking,
            check_thanh_toan_truoc, check_khong_can_thanh_toan_truoc,
            check_huy_mien_phi, check_khong_can_the_tin_dung,
            check_linh_dong_doi_ngay, check_khong_hoan_tien, check_bao_gom_bua_sang
        ]

        strict_score = sum(strict_checks)

        # Trừ điểm nếu check-in/check-out/giờ giới nghiêm sai
        if not check_time_in:
            strict_score -= 3 
        if not check_time_out:
            strict_score -= 3 
        if not check_limit_time:
            strict_score -= 0.5 

        final_score = strict_score + total_similarity
        # Scale về khoảng [0, 1]
        max_possible_score = len(strict_checks) + len(shared_keys)
        final_score = max(0.0, min(final_score / max_possible_score, 1.0))

        return final_score

