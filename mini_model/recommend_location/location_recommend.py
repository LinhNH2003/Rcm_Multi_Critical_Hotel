import os
import json
import pandas as pd
import numpy as np
import unicodedata
from difflib import SequenceMatcher
import re
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import wikipedia
import requests
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from math import radians, sin, cos, sqrt, atan2
from math import log1p


import sys
from pathlib import Path

# Thêm thư mục gốc (mini_model) vào sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config


# Import vietnam location data
neighbor_province = pd.read_csv(config.get_path('neighbor_province.csv'))
vn_location = pd.read_excel(config.get_path('vietnam_location.xlsx'),sheet_name = 'Sheet1')
vietnam_province = vn_location["Tỉnh Thành Phố"].unique()


def transform_near_places(data):
    """
    Transforms a list of nearby places with distance information into a standardized format.
    
    Args:
        data (list of dict): A list where each dictionary represents a category of places with
                             'title' as the category name and 'detail' as a dictionary of places
                             and their respective distances.

    Returns:
        list of dict: A list where each dictionary contains 'place', 'distance', and 'category'.
                      The 'distance' is converted to a float in kilometers. If conversion fails,
                      'distance' is set to None.
    """

    transformed_data = []
    for item in data:
        title = item['title']
        details = item['detail']
        for place, distance in details.items():
            # Xử lý distance
            if isinstance(distance, str):
                # Loại bỏ 'km' và khoảng trắng
                distance_str = distance.replace(' km', '').strip()
                # Thay dấu phẩy bằng dấu chấm để chuyển thành float
                distance_str = distance_str.replace(',', '.')
                try:
                    distance_float = float(distance_str)
                except ValueError:
                    distance_float = None  # Nếu không chuyển được (trường hợp lỗi)
            else:
                distance_float = float(distance)  # Nếu distance đã là số
            
            transformed_data.append({
                'place': place,
                'distance': distance_float,
                'category': title  # Giữ lại category nếu cần
            })
    return transformed_data


### **Find best location hotel**

def dms_to_decimal(dms_str):
    """
    Chuyển tọa độ từ định dạng DMS (độ, phút, giây) sang số thập phân (DD).
    Hỗ trợ cả Bắc (B), Nam (S), Đông (Đ), Tây (W).
    """
    # Chuẩn hóa chuỗi (đổi dấu phẩy thành dấu chấm nếu có)
    dms_str = dms_str.replace(',', '.')

    # Biểu thức chính quy để tách độ, phút, giây và hướng (có thể thiếu phút hoặc giây)
    match = re.match(r"(\d+)°(\d+)?′?([\d\.]+)?″?([BĐNSEW])", dms_str)
    if not match:
        raise ValueError(f"Định dạng không hợp lệ: {dms_str}")
    
    degrees, minutes, seconds, direction = match.groups()
    
    # Chuyển None thành 0 nếu không có phút hoặc giây
    degrees = int(degrees)
    minutes = int(minutes) if minutes else 0
    seconds = float(seconds) if seconds else 0.0
    
    # Chuyển đổi sang số thập phân
    decimal = degrees + minutes / 60 + seconds / 3600

    # Xác định dấu âm cho Tây (W) và Nam (S)
    if direction in ["S", "W"]:
        decimal *= -1
    
    return round(decimal, 6)

def convert_lat_lon(value):
    """Kiểm tra nếu value là chuỗi DMS thì chuyển đổi, nếu không giữ nguyên."""
    if isinstance(value, str):
        try:
            return dms_to_decimal(value)
        except ValueError:
            return value  # Nếu không đúng định dạng DMS thì giữ nguyên
    return value  # Nếu đã là số, giữ nguyên



# Khởi tạo Geocoder
geolocator = Nominatim(user_agent="airport_locator")
# Hàm lấy tọa độ
def get_lat_lon(location):

    location_data = geolocator.geocode(location, timeout=10)
    if location_data:
        return location_data.latitude, location_data.longitude
    else:
        return None, None



def get_lat_lng_wiki(place_name, lang='vi'):
    wikipedia.set_lang(lang)
    
    try:
        search_results = wikipedia.search(place_name)
        if not search_results:
            return None, None  
        
        # Lấy trang đầu tiên trong danh sách kết quả
        page_title = max(search_results, key=lambda title: SequenceMatcher(None, place_name, title).ratio())
        page = wikipedia.page(page_title, auto_suggest=False)
        
        # Lấy mã nguồn HTML của trang Wikipedia
        response = requests.get(page.url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Tìm thẻ chứa tọa độ
        lat_span = soup.find("span", class_="latitude")
        lng_span = soup.find("span", class_="longitude")

        if lat_span and lng_span:
            lat = convert_lat_lon(lat_span.text)
            lng = convert_lat_lon(lng_span.text)
            return lat, lng
        else:
            return  None, None 
    except Exception as e:
        print(f"Lỗi: {e}")
        return None, None

# Danh sách tên các tỉnh/thành phố chuẩn của Việt Nam (lấy từ df_station['Tỉnh/ Tp'].unique())
STANDARD_PROVINCES = [
    "An Giang", "Bà Rịa - Vũng Tàu", "Bắc Giang", "Bắc Kạn", "Bạc Liêu", "Bắc Ninh",
    "Bến Tre", "Bình Định", "Bình Dương", "Bình Phước", "Bình Thuận", "Cà Mau",
    "Cần Thơ", "Cao Bằng", "Đà Nẵng", "Đắk Lắk", "Đắk Nông", "Điện Biên", "Đồng Nai",
    "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Nội", "Hà Tĩnh", "Hải Dương",
    "Hải Phòng", "Hậu Giang", "Hòa Bình", "Hưng Yên", "Khánh Hòa", "Kiên Giang",
    "Kon Tum", "Lai Châu", "Lâm Đồng", "Lạng Sơn", "Lào Cai", "Long An", "Nam Định",
    "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên", "Quảng Bình",
    "Quảng Nam", "Quảng Ngãi", "Quảng Ninh", "Quảng Trị", "Sóc Trăng", "Sơn La",
    "Tây Ninh", "Thái Bình", "Thái Nguyên", "Thanh Hóa", "Thừa Thiên Huế", "Tiền Giang",
    "TP Hồ Chí Minh", "Trà Vinh", "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái"
]

def normalize_province(province_name):
    """
    Chuẩn hóa tên tỉnh/thành phố về dạng chuẩn.
    Nếu không tìm thấy tỉnh phù hợp, trả về chính tên đầu vào.
    """
    if not isinstance(province_name, str) or not province_name.strip():
        return None  # Trả về None nếu đầu vào không hợp lệ

    # Chuẩn hóa chuỗi đầu vào
    cleaned_name = province_name.lower().strip()
    cleaned_name = cleaned_name.replace("tp.", "").replace("thành phố", "").replace("tỉnh", "").strip()

    # Xử lý các trường hợp đặc biệt trước
    if "phú quốc" in cleaned_name:
        return 'Kiên Giang'
    if "hcm" in cleaned_name or "hồ chí minh" in cleaned_name:
        return "TP Hồ Chí Minh"
    if "hn" in cleaned_name or "hà nội" in cleaned_name:
        return "Hà Nội"
    if "đn" in cleaned_name or "đà nẵng" in cleaned_name:
        return "Đà Nẵng"
    # Còn lại 
    province = max(STANDARD_PROVINCES, key=lambda title: SequenceMatcher(None, cleaned_name, title).ratio())

    return province 

def extract_address(address):
    address = unicodedata.normalize("NFC", address)
    address = address.replace("Ð", "Đ")
    address = re.sub(r",?\s*(Việt Nam|VN|việt nam|vn|Vn)", "", address, flags=re.IGNORECASE).strip()

    # Quận/Huyện và Tỉnh/TP
    district_pattern = re.compile(r'\b(Quận|Huyện|Thị xã|TP\.|Thành phố)\s*\.?\s*(\d+|[^\d,]+)', re.IGNORECASE)
    city_pattern = re.compile(r'\b(TP\.|Thành phố|Tỉnh|tp\.)\s*\.?\s*([^\d,]+)', re.IGNORECASE)

    # Tìm quận/huyện
    district_match = district_pattern.search(address)
    district = district_match.group(0) if district_match else None

    # Tìm tỉnh/thành phố
    city_match = city_pattern.search(address)
    city = city_match.group(0) if city_match else None

    # Kiểm tra nếu là đảo hoặc cù lao
    island_match = re.search(r'\b(Đảo|Cù Lao)\b', address, re.IGNORECASE)
    island = island_match.group(0) if island_match else None

    # Kiểm tra trường hợp đặc biệt
    lower_address = address.lower()
    if "đảo cát bà" in lower_address:
        city = "Hải Phòng"
        district = "Huyện Cát Hải"
        island = "Đảo Cát Bà"
    elif "hồ tràm" in lower_address or "ho tram" in lower_address:
        city = "Bà Rịa - Vũng Tàu"
        district = "Huyện Xuyên Mộc"
    
    if city is None:
        parts = [" ".join(part.strip().split()) for part in address.split(",") if part.strip()]
        check = parts[-1].lower()

        # Chuẩn hóa dữ liệu trong `vn_location`
        vn_location["Quận Huyện Lower"] = vn_location["Quận Huyện"].str.lower()
        vn_location["Tỉnh Thành Phố Lower"] = vn_location["Tỉnh Thành Phố"].str.lower()

        city_check = vn_location[vn_location["Tỉnh Thành Phố Lower"].str.contains(check, case=False, na=False, regex=False)]
        if not city_check.empty:
            city = city_check["Tỉnh Thành Phố"].values[0]
            district = parts[-2] if len(parts) > 1 else None
        else:
            matched_row = vn_location[vn_location["Quận Huyện Lower"].str.contains(check, case=False, na=False, regex=False)]
            if not matched_row.empty:
                district = matched_row["Quận Huyện"].values[0]
                city = matched_row["Tỉnh Thành Phố"].values[0]
            else:
                district = None
                city = None

    return district, normalize_province(city), island


def is_similar(query, text, threshold=80):
    """Kiểm tra xem text có giống với query theo ngưỡng xác suất hay không"""
    return fuzz.partial_ratio(query.lower(), text.lower()) >= threshold


### **Find Hotel**
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import numpy as np


def find_hotels_near_location(data_dict, places, location, max_distance_km=20):
    """
    Finds and returns a list of hotels near specified tourist locations from a dictionary-based dataset.

    Args:
        data_dict (dict): A dictionary of hotels with hotel_id as key and info as value.
        places (list): A list of tourist locations to search nearby.
        location (str): The location used when places is empty, to filter by province.
        max_distance_km (int, optional): Maximum distance (in kilometers) from the centroid of the tourist places.

    Returns:
        list: A list of unique hotel info dictionaries near the specified places.
    """
    geolocator = Nominatim(user_agent="hotel_locator")
    places_location = []

    if not places:
        relevant_provinces = set([normalize_province(location)])
        return [
            hotel for hotel in data_dict.values()
            if normalize_province(hotel.get('province/ city', '')) in relevant_provinces
        ]

    relevant_provinces = set()
    
    for place in places:
        loc = geolocator.geocode(place, timeout=10)
        if loc:
            province = extract_address(loc.address)[1]
            if province:
                relevant_provinces.update(neighbor_province.get(province, []) + [province])
                places_location.append((loc.latitude, loc.longitude))

    if not places_location:
        return []

    relevant_hotels = [
        hotel for hotel in data_dict.values()
        if normalize_province(hotel.get('province/ city', '')) in relevant_provinces
    ]

    matched_hotels = []

    for hotel in relevant_hotels:
        nearby_places = hotel.get('nearby_places', [])
        description = hotel.get('description', "").lower()
        for user_place in places:
            if user_place.lower() in description:
                matched_hotels.append(hotel)
                break
            else:
                found = False
                for place in nearby_places:
                    for place_name, _ in place.get('detail', {}).items():
                        if user_place.lower() in str(place_name).lower() or is_similar(user_place, str(place_name)):
                            matched_hotels.append(hotel)
                            found = True
                            break
                    if found:
                        break

    if len(matched_hotels) <= 50:
        centroid = np.mean(places_location, axis=0)
        max_place_distance = max(float(geodesic(centroid, place).km) for place in places_location)
        radius = max(float(max_distance_km), float(max_place_distance))

        existing_ids = set(hotel['id'] for hotel in matched_hotels)

        filtered_hotels = [
            hotel for hotel in relevant_hotels
            if hotel['id'] not in existing_ids and
               geodesic(
                   centroid,
                   tuple(map(float, hotel.get('lat_lng', (0, 0))))
               ).km <= radius
        ]
        matched_hotels.extend(filtered_hotels)

    unique_hotels = {hotel['id']: hotel for hotel in matched_hotels}.values()
    return list(unique_hotels)



### **Location Score**
def clean_distance(distance):
    """
    Chuẩn hóa khoảng cách từ string (vd: '350 m', '5 km') hoặc số -> float (km)
    """
    if isinstance(distance, (int, float)):
        return float(distance)
    
    # Loại bỏ khoảng trắng thừa và chuyển đổi dấu phẩy thành chấm
    distance = distance.strip().replace(',', '.')
    
    # Tách số và đơn vị
    if ' ' in distance:
        num_part, unit = distance.split(' ', 1)
    else:
        # Trường hợp không có đơn vị (mặc định là km)
        num_part, unit = distance, 'km'
    
    try:
        num = float(num_part)
    except ValueError:
        return None  # Không thể chuyển đổi
    
    # Chuyển đổi sang km
    unit = unit.lower()
    if unit == 'm':
        return num / 1000  # m -> km
    elif unit == 'km':
        return num
    else:
        return None 

# def calculate_location_score(hotel, user_places, province=None, is_near_center=False):
#     """
#     Tính điểm vị trí khách sạn (0-100) dựa trên:
#     - Độ phủ điểm đến user quan tâm
#     - Tổng khoảng cách đến các điểm
#     - Mật độ tiện ích xung quanh
#     - Ưu tiên gần trung tâm (nếu có)
#     """
#     nearby_places = hotel.get('nearby_places', [])
#     total_score = 0
    
#     # 1. Chuyển đổi nearby_places sang dict {place: distance} và làm sạch distance
#     places_data = {}
#     for category in nearby_places:
#         for place, distance in category['detail'].items():
#             # Làm sạch distance (chuyển về km dạng float)
#             dist_km = clean_distance(distance)
#             places_data[place] = dist_km
    
#     # 2. Tính độ phủ điểm đến user (30 điểm)
#     matched_places = []
#     for user_place in user_places:
#         for place in places_data.keys():
#             if is_similar(user_place, place):
#                 matched_places.append(place)
#                 break
    
#     coverage_score = (len(matched_places) / len(user_places)) * 30 if user_places else 0
    
#     # 3. Tính điểm tổng khoảng cách (30 điểm)
#     distance_score = 0
#     if matched_places:
#         total_distance = sum(places_data[place] for place in matched_places)
#         # Chuẩn hóa điểm: khoảng cách trung bình <5km = 30đ, <10km = 20đ, <20km = 10đ
#         avg_distance = total_distance / len(matched_places)
#         distance_score = max(0, 30 - (avg_distance / 5))  # Giảm 6 điểm mỗi 5km
    
#     # 4. Tính mật độ tiện ích (20 điểm)
#     amenities_count = len(places_data)
#     amenities_score = min(20, amenities_count * 0.5)  # Mỗi địa điểm = 0.5đ, tối đa 20đ
    
#     # 5. Kiểm tra gần trung tâm (20 điểm)
#     center_score = 0
#     if is_near_center and province:
#         # Kiểm tra xem có địa điểm chứa "trung tâm" hoặc tên tỉnh không
#         center_keywords = ["trung tâm", "city center", "central", province.split()[0]]
#         for place in places_data.keys():
#             if any(keyword.lower() in place.lower() for keyword in center_keywords):
#                 if places_data[place] <= 5:  # <=5km từ trung tâm
#                     center_score = 20
#                 elif places_data[place] <= 10:  # <=10km
#                     center_score = 10
#                 break
    
#     # Tổng hợp điểm
#     total_score = coverage_score + distance_score + amenities_score + center_score
#     #return min(100, round(total_score, 2))
#     return round(min(100, total_score) / 100, 4)

def calculate_location_score(hotel, user_places, province=None, is_near_center=False):
    """
    Tính điểm vị trí khách sạn [0.0 - 1.0] dựa trên:
    - Độ phủ điểm đến user quan tâm
    - Tổng khoảng cách đến các điểm
    - Mật độ tiện ích xung quanh
    - Ưu tiên gần trung tâm (dùng hàm log)
    """
    nearby_places = hotel.get('nearby_places', [])
    
    # 1. Làm sạch: chuyển nearby_places thành dict {place: distance_km}
    places_data = {}
    for category in nearby_places:
        for place, distance in category['detail'].items():
            dist_km = clean_distance(distance)  # Hàm này cần trả về float (km)
            places_data[place] = dist_km
    
    # 2. Điểm độ phủ [0-1]
    matched_places = []
    for user_place in user_places:
        for place in places_data:
            if is_similar(user_place, place):  # Cần định nghĩa is_similar
                matched_places.append(place)
                break
    coverage_score = len(matched_places) / len(user_places) if user_places else 0.0

    # 3. Điểm khoảng cách [0-1]
    distance_score = 0.0
    if matched_places:
        avg_distance = sum(places_data[p] for p in matched_places) / len(matched_places)
        max_distance = 20  # >20km thì 0 điểm
        distance_score = max(0.0, 1 - (avg_distance / max_distance))

    # 4. Điểm mật độ tiện ích [0-1]
    amenities_count = len(places_data)
    amenities_score = min(1.0, log1p(amenities_count) / log1p(30))  # Chuẩn hóa theo mốc 30 tiện ích

    # 5. Điểm gần trung tâm [0-1] tính trực tiếp trong hàm
    center_score = 0.0
    if is_near_center and province:
        center_keywords = ["trung tâm", "city center", "central", province.split()[0]]
        for place in places_data:
            if any(k.lower() in place.lower() for k in center_keywords):
                dist = places_data[place]
                max_dist = 20
                if dist > max_dist:
                    center_score = 0.0
                else:
                    center_score = max(0.0, 1 - log1p(dist) / log1p(max_dist))
                break

    # 6. Tổng điểm = trung bình các thành phần
    total_score = (coverage_score + distance_score + amenities_score + center_score) / 4.0
    return round(total_score, 4)

def get_location_score(data_list, user_places, province=None, is_near_center=False):
    """
    Thêm trường location_score vào mỗi khách sạn trong data_list
    """
    for hotel in data_list:
        score = calculate_location_score(
            hotel,
            user_places,
            province,
            is_near_center
        )
        hotel['location_score'] = score
    return sorted(data_list, key=lambda x: x['location_score'], reverse=True)


