import os
import json
import pandas as pd
import numpy as np
import unicodedata
from difflib import SequenceMatcher
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from datetime import datetime

#### **room_service_included**
# Biểu thức chính quy để trích xuất thông tin
patterns = {
    "thanh_toan_truoc": r"\nThanh toán trước[^\n]*",
    "khong_can_thanh_toan_truoc": r"Không cần thanh toán trước - thanh toán tại chỗ nghỉ",
    "huy_mien_phi": r"(Hủy miễn phí trước \d{1,2} tháng \d{1,2}, \d{4})|(Miễn phí hủy đến.*? \d{1,2} tháng \d{1,2}, \d{4})",
    "khong_can_the_tin_dung": r"Không cần thẻ tín dụng",
    "linh_dong_doi_ngay": r"Linh động đổi ngày khi kế hoạch thay đổi",
    "khong_hoan_tien": r"Không hoàn tiền",
    "bao_gom_bua_sang": r"(Bao bữa sáng|Bao gồm bữa sáng|Bữa sáng [^\n]*|Giá bao gồm bữa sáng)"
}


# Hàm trích xuất thông tin
def extract_room_service_features(text):
    """
    Trích xuất các thông tin về dịch vụ phòng, bao gồm:

    * Thanh toán trước
    * Không cần thanh toán trước
    * Hủy miễn phí
    * Không cần thẻ tín dụng
    * Linh động đổi ngày
    * Không hoàn tiền
    * Bao bữa sáng

    Output: dict với các key là tên các dịch vụ và value là True/False
    """
    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            extracted[key] = True
        else:
            extracted[key] = False
    return extracted

# Hàm trích xuất thông tin
def extract_features_string(text):
    """
    Extracts specific features from a given text string based on predefined patterns.

    Args:
        text (str): The input string from which features are to be extracted.

    Returns:
        dict: A dictionary with keys as feature names and values as the matched
              feature strings from the input text. If a pattern is not matched,
              the feature is not included in the dictionary.
    """

    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            extracted[key] = match.group(0)
    return extracted


#### **Độ tuổi**
def extract_age_range(text):
    """
    Extracts an age range from a given text string.

    Args:
        text (str): A string potentially containing age range information.

    Returns:
        tuple: A tuple of two integers representing the age range (start_age, end_age).
               If only a starting age is found, returns (start_age, None).
               If only an ending age is found, returns (None, end_age).
               If no age information is found, returns (None, None).

    The function handles different patterns to extract age ranges, including:
        - "từ X - Y tuổi" for a range
        - "từ X tuổi đến Y tuổi" for a range
        - "từ X tuổi" for a starting age
        - "đến Y tuổi" for an ending age
        - "độ tuổi tối thiểu", "tối thiểu", or "độ tuổi thấp nhất" for a minimum age
    """

    if pd.isna(text) or not isinstance(text, str):
        return (None, None)

    text = text.lower()

    # Trường hợp từ X - Y tuổi
    match_range = re.search(r'từ (\d{1,3})\s*-\s*(\d{1,3}) tuổi', text)
    if match_range:
        return (int(match_range.group(1)), int(match_range.group(2)))

    # Trường hợp từ X tuổi đến Y tuổi
    match_from_to = re.search(r'từ (\d{1,3}) tuổi.*?đến (\d{1,3}) tuổi', text)
    if match_from_to:
        return (int(match_from_to.group(1)), int(match_from_to.group(2)))

    # Trường hợp chỉ có từ X tuổi
    match_from = re.search(r'từ (\d{1,3}) tuổi', text)
    if match_from:
        return (int(match_from.group(1)), None)

    # Trường hợp chỉ có đến Y tuổi
    match_to = re.search(r'đến (\d{1,3}) tuổi', text)
    if match_to:
        return (None, int(match_to.group(1)))

    # Trường hợp các cụm: độ tuổi tối thiểu, tối thiểu, độ tuổi thấp nhất
    match_min = re.search(r'(độ tuổi tối thiểu|tối thiểu|độ tuổi thấp nhất).*?(\d{1,3})', text)
    if match_min:
        return (int(match_min.group(2)), None)

    return (None, None)
def extract_child_policy_info(text):
    """
    Extracts information about allowed children ages from a given text string.

    Args:
        text (str): A string potentially containing information about allowed children ages.

    Returns:
        tuple: A tuple of two integers representing the allowed age range (min_age, max_age).
               If no age information is found, returns (0, 18), indicating all children under 18 are allowed.

    The function handles different patterns to extract allowed age ranges, including:
        - "từ X - Y tuổi" for a range
        - "từ X tuổi trở lên" for a minimum age
        - "X tuổi" for a specific age
    """
    if not isinstance(text, str):
        return None

    # Biểu thức chính quy để tìm độ tuổi trẻ em cho phép
    age_pattern = r'\b(\d{1,3})\s*-\s*(\d{1,3})\s*tuổi\b|\b(\d{1,3})\s*tuổi trở lên\b|\b(\d{1,3})\s*tuổi\b'
    age_match = re.findall(age_pattern, text)
    
    # Xác định độ tuổi trẻ em cho phép
    allowed_age = (0, 18)  # Mặc định là tất cả trẻ em dưới 18 tuổi
    for match in age_match:
        if match[0] and match[1]:  # Nếu có khoảng tuổi (min - max)
            allowed_age = (int(match[0]), int(match[1]))
        elif match[2]:  # Nếu có độ tuổi tối thiểu
            allowed_age = (int(match[2]), 18)  # Giới hạn tối đa là 18
        elif match[3]:  # Nếu chỉ có độ tuổi
            allowed_age = (int(match[3]), 18)

    return (allowed_age)



def extract_time_range(text):
    """
    Extracts a time range from a given text string.

    Args:
        text (str): A string potentially containing time range information.

    Returns:
        tuple: A tuple of two strings representing the extracted time range (start_time, end_time).
               If no time range is found, returns None.

    The function handles different patterns to extract time ranges, including:
        - "từ X - Y giờ" for a range
        - "từ X giờ đến Y giờ" for a range
        - "từ X giờ" for a starting time
        - "đến Y giờ" for an ending time
        - "X giờ" for a single time
    """

    if not isinstance(text, str):
        return None

    text_lower = text.lower().strip()

    # Trường hợp phục vụ 24h
    if "24 giờ" in text_lower or "24/24" in text_lower or "cả ngày" in text_lower or "không có giờ giới hạn" in text_lower:
        return ("00:00", "24:00")

    # Tìm định dạng dạng "10:00-12:00" hoặc "10:00 - 12:00"
    dash_pattern = r'\b(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\b'
    dash_match = re.search(dash_pattern, text)
    if dash_match:
        try:
            time1 = datetime.strptime(dash_match.group(1).zfill(5), "%H:%M").strftime("%H:%M")
            time2 = datetime.strptime(dash_match.group(2).zfill(5), "%H:%M").strftime("%H:%M")
            return (time1, time2)
        except ValueError:
            return None

    # Tìm tất cả thời gian
    time_pattern = r'\b(\d{1,2}:\d{2})\b'
    times = re.findall(time_pattern, text)

    # Chuẩn hóa giờ
    try:
        time_objs = [datetime.strptime(t.zfill(5), "%H:%M") for t in times]
        times_standardized = [t.strftime("%H:%M") for t in time_objs]
    except ValueError:
        return None

    # Trích xuất khoảng thời gian dựa trên văn cảnh
    if "từ" in text_lower and "đến" in text_lower and len(times_standardized) >= 2:
        time_range = (times_standardized[0], times_standardized[1])
    elif "từ" in text_lower and len(times_standardized) >= 1:
        start = min(times_standardized)
        end = max(times_standardized) if len(times_standardized) > 1 else "24:00"
        time_range = (start, end)
    elif "đến" in text_lower and len(times_standardized) >= 1:
        time_range = ("00:00", times_standardized[0])
    elif len(times_standardized) == 1:
        time_range = (times_standardized[0], "24:00")
    else:
        time_range = None

    return time_range

from datetime import datetime, timedelta

def time_to_minutes(t):
    """Chuyển đổi thời gian 'HH:MM' thành số phút kể từ 00:00."""
    h, m = map(int, t.split(':'))
    return h * 60 + m

def is_time_range_valid(time_range_to_check, valid_range):
    """
    Kiểm tra khoảng thời gian `time_range_to_check` có nằm hoàn toàn trong `valid_range` không.
    Cả hai đều là tuple dạng ('HH:MM', 'HH:MM')
    """
    start_check = time_to_minutes(time_range_to_check[0])
    end_check = time_to_minutes(time_range_to_check[1])
    start_valid = time_to_minutes(valid_range[0])
    end_valid = time_to_minutes(valid_range[1])

    def in_range(start, end, point_start, point_end):
        # Xử lý thời gian qua đêm (ví dụ: 23:00 - 05:00)
        if end < start:
            end += 1440  # cộng thêm 24 giờ
        if point_start < start:
            point_start += 1440
        if point_end <= point_start:
            point_end += 1440
        return start <= point_start and point_end <= end

    return in_range(start_valid, end_valid, start_check, end_check)

#### **Thanh toán**
def extract_deposit_amount(text):
    """
    Extracts the deposit amount in VND from a given text string.

    Args:
        text (str): A string potentially containing deposit amount information.

    Returns:
        int: The extracted deposit amount in VND. If no amount is found, returns None.

    The function handles different patterns to extract deposit amounts, including:
        - "VND [amount]"
        - "VND.[amount]"
    """
    match = re.search(r'VND\s*([\d\.]+)', text)
    if match:
        # Loại bỏ dấu chấm và chuyển về kiểu int
        amount_str = match.group(1).replace('.', '')
        return int(amount_str)
    return None


#### **Ưu đãi và dịch vụ kèm theo**
def check_pet_policy(policy_text):
    """
    Checks if a given policy text allows pets and if it's free.

    Args:
        policy_text (str): The policy text to check.

    Returns:
        list: A list of two boolean values, where the first indicates if pets are allowed
              and the second indicates if it's free.

    The function handles different patterns to detect if pets are allowed and if it's free, including:
        - "Vật nuôi được phép" or "Vật nuôi được mang" for allowing pets
        - "Không tính thêm phí" or "Miễn phí" for not charging extra
        - "Không cho phép vật nuôi" for not allowing pets
        - "Không miễn phí" for not being free
    """
    if not isinstance(policy_text, str):
        return [False, True]

    text = policy_text.lower()

    # Kiểm tra cho phép vật nuôi
    allow_pet = bool(
        re.search(r"(vật nuôi được phép|cho phép mang|có thể mang)", text)
        and not re.search(r"không\s+(cho phép|được phép).*vật nuôi", text)
    )

    # Kiểm tra miễn phí / không phụ thu
    free_fee = bool(
        (re.search(r"không\s+tính thêm phí", text) or
         "miễn phí" in text or "không phụ thu" in text)
        and "không miễn phí" not in text
    )

    return [allow_pet, free_fee]

# Tính toán score policies 
def compute_tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]


def find_similar_hotel_policies(user_input, hotel): 
    hotel_policies = hotel['policies_hotels']
    
    # 1. Check-in/out
    check_time_in = check_time_out = True
    hotel_checkin = extract_time_range(hotel_policies.get('Nhận phòng'))
    user_checkin = extract_time_range(user_input.get('Nhận phòng'))
    if hotel_checkin and user_checkin:
        check_time_in = is_time_range_valid(user_checkin, hotel_checkin)

    hotel_checkout = extract_time_range(hotel_policies.get('Trả phòng'))
    user_checkout = extract_time_range(user_input.get('Trả phòng'))
    if hotel_checkout and user_checkout:
        check_time_out = is_time_range_valid(user_checkout, hotel_checkout)

    # 2. Giờ giới nghiêm
    check_limit_time = True
    if user_input.get('Giờ giới nghiêm') and hotel_policies.get('Giờ giới nghiêm'):
        user_limit_time = extract_time_range(user_input['Giờ giới nghiêm'])
        hotel_limit_time = extract_time_range(hotel_policies['Giờ giới nghiêm'])
        if user_limit_time and hotel_limit_time:
            check_limit_time = is_time_range_valid(user_limit_time, hotel_limit_time)

    # 3. Giờ yên lặng
    check_silent_time = True
    if user_input.get('Thời gian yên lặng') and hotel_policies.get('Thời gian yên lặng'):
        user_silent_time = extract_time_range(user_input['Thời gian yên lặng'])
        hotel_silent_time = extract_time_range(hotel_policies['Thời gian yên lặng'])
        if user_silent_time and hotel_silent_time:
            check_silent_time = is_time_range_valid(user_silent_time, hotel_silent_time)

    # 4. Độ tuổi
    check_age = True
    if hotel_policies.get('Giới hạn độ tuổi'):
        hotel_age_range = extract_age_range(hotel_policies['Giới hạn độ tuổi'])
        user_child_age = extract_child_policy_info(user_input.get('Trẻ em và giường', '0 tuổi'))
        hotel_child_age = extract_child_policy_info(hotel_policies.get('Trẻ em và giường', '0 tuổi'))
        check_age = user_child_age <= (hotel_child_age if hotel_child_age else hotel_age_range[0])

    # 5. Đặt cọc
    check_deposit = True
    if user_input.get('Đặt cọc đề phòng hư hại có thể hoàn lại') and hotel_policies.get('Đặt cọc đề phòng hư hại có thể hoàn lại'):
        user_deposit = extract_deposit_amount(user_input['Đặt cọc đề phòng hư hại có thể hoàn lại'])
        hotel_deposit = extract_deposit_amount(hotel_policies['Đặt cọc đề phòng hư hại có thể hoàn lại'])
        if hotel_deposit and user_deposit and hotel_deposit > user_deposit:
            check_deposit = False

    # 6. Tiệc tùng
    check_party = True
    if user_input.get('Tiệc tùng'):
        if hotel_policies.get('Tiệc tùng') and re.search('Không cho phép', hotel_policies['Tiệc tùng']):
            check_party = False

    # 7. Vật nuôi
    check_pet = check_pet_miscellenous = True
    if user_input.get('Vật nuôi') and hotel_policies.get('Vật nuôi'):
        user_pet = check_pet_policy(user_input['Vật nuôi'])
        hotel_pet = check_pet_policy(hotel_policies['Vật nuôi'])
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
        if hotel_policies.get('Hút thuốc') and re.search('Không cho phép hút thuốc', hotel_policies['Hút thuốc']):
            if not re.search('Không cho phép hút thuốc', user_input['Hút thuốc']):
                check_smoking = False

    # 11. TF-IDF similarity
    shared_keys = set(user_input.keys()).intersection(set(hotel_policies.keys()))
    total_similarity = 0
    weight = 0 

    for key in shared_keys:
        #print(key)
        user_val = user_input.get(key)
        hotel_val = hotel_policies.get(key)
        if isinstance(user_val, str) and isinstance(hotel_val, str):
            sim = compute_tfidf_similarity(user_val, hotel_val)
            weight+=1 
            total_similarity += sim

    strict_checks_1 = [
        check_time_in, check_time_out, check_age, check_deposit,
        check_limit_time, check_silent_time, check_party, check_pet, check_group, check_smoking

    ]
    strict_checks_2 = [check_pet_miscellenous] 

    # Tính tổng số điểm tối đa
    max_score = len(strict_checks_1) + weight + 1 

    # Tính điểm kiểm tra nghiêm ngặt
    if check_pet_miscellenous:
        strict_score = sum(strict_checks_1) 
    else:
        strict_score = sum(strict_checks_1) + 1 

    # Trừ điểm nếu vi phạm quan trọng
    if not check_time_in:
        strict_score -= 3 
    if not check_time_out:
        strict_score -= 3 
    if not check_limit_time:
        strict_score -= 0.5 

    #print(strict_score, total_similarity)

    final_score = strict_score + total_similarity

    # Chuẩn hóa điểm về khoảng [0, 1]
    normalized_score = round(final_score / max_score, 4) if max_score > 0 else 0.0

    return normalized_score