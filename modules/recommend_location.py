import re
import requests
import unicodedata
import math
import wikipedia

from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from rapidfuzz import fuzz
from geopy.geocoders import Nominatim
from typing import Optional, Tuple, Dict, Any, List


class PlaceTransformer:
    def __init__(self, unit: str = "km"):
        self.unit = unit

    def transform_near_places(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chuẩn hóa cấu trúc nearby_places -> list[{place, distance(float|None), category}]
        data: [{"title": str, "detail": {place: distance(str|float|int), ...}}, ...]
        """
        transformed_data: List[Dict[str, Any]] = []
        if not isinstance(data, list):
            return transformed_data

        for item in data:
            title = item.get('title')
            details = item.get('detail', {})
            if not isinstance(details, dict) or not title:
                continue

            for place, distance in details.items():
                if isinstance(distance, str):
                    # loại bỏ " km" theo đơn vị hiện tại, đổi , -> .
                    distance_str = (
                        distance.replace(f" {self.unit}", "")
                        .replace(self.unit, "")
                        .strip()
                        .replace(",", ".")
                    )
                    # giữ lại phần số (trường hợp "1.2km", "850 m", "khoảng 2 km")
                    m = re.search(r"[-+]?\d*\.?\d+", distance_str)
                    if m:
                        try:
                            distance_float = float(m.group(0))
                            # nếu đơn vị là mét trong chuỗi gốc -> đổi về km
                            if re.search(r"\bm\b", distance.lower()):
                                distance_float = distance_float / 1000.0
                        except ValueError:
                            distance_float = None
                    else:
                        distance_float = None
                else:
                    try:
                        distance_float = float(distance)
                    except (TypeError, ValueError):
                        distance_float = None

                transformed_data.append({
                    "place": str(place),
                    "distance": distance_float,
                    "category": str(title)
                })
        return transformed_data


class CoordinateConverter:
    """Xử lý chuyển đổi tọa độ DMS <-> Decimal."""
    def __init__(self, precision: int = 6):
        self.precision = precision

    def dms_to_decimal(self, dms_str: str) -> float:
        if not isinstance(dms_str, str):
            raise ValueError(f"Định dạng không hợp lệ: {dms_str}")

        dms_str = dms_str.strip().replace(",", ".")
        # Hỗ trợ ký tự phút/giây phổ biến: ′ ' ″ "
        match = re.match(
            r"^\s*(\d+)°\s*(\d+)?(?:′|')?\s*([\d\.]+)?(?:″|\")?\s*([BĐNSEW])\s*$",
            dms_str
        )
        if not match:
            raise ValueError(f"Định dạng không hợp lệ: {dms_str}")

        degrees, minutes, seconds, direction = match.groups()
        degrees = int(degrees)
        minutes = int(minutes) if minutes else 0
        seconds = float(seconds) if seconds else 0.0

        decimal = degrees + minutes / 60.0 + seconds / 3600.0

        # Hướng: B(ắc)=N, N(am)=S, Đ(ông)=E, Tây=W; cũng hỗ trợ N S E W
        if direction in ["S", "W", "N".lower(), "E".lower()]:  # giữ nguyên nhánh S/W
            pass  # giữ cho rõ ràng
        # map tiếng Việt
        if direction == "N":  # Nam -> âm (giống S)
            decimal *= -1
        if direction == "W":  # Tây -> âm (giống W)
            decimal *= -1
        if direction == "B":  # Bắc -> dương
            decimal = decimal
        if direction == "Đ":  # Đông -> dương
            decimal = decimal

        # Chuẩn: chỉ S và W mang dấu âm; B/Đ/N/E/W đã xử lý ở trên, nhưng đảm bảo:
        if direction in ["S", "W"]:
            decimal = -abs(decimal)
        elif direction in ["B", "Đ", "N", "E"]:
            decimal = abs(decimal)

        return round(decimal, self.precision)

    def convert_lat_lon(self, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return self.dms_to_decimal(value)
            except ValueError:
                return value
        return value


class GeoLocator:
    """Dịch vụ lấy tọa độ bằng geopy hoặc Wikipedia."""

    def __init__(self, user_agent: str = "hotel_locator", coord_precision: int = 6):
        self.geolocator = Nominatim(user_agent=user_agent)
        self.coord = CoordinateConverter(precision=coord_precision)

    def get_lat_lon(self, location: str) -> Tuple[Optional[float], Optional[float]]:
        """Lấy lat/lon từ Nominatim (OpenStreetMap)."""
        try:
            loc = self.geolocator.geocode(location, timeout=10)
            if loc:
                return float(loc.latitude), float(loc.longitude)
        except Exception as e:
            print(f"Geopy error: {e}")
        return None, None

    def get_lat_lng_wiki(self, place_name: str, lang: str = "vi") -> Tuple[Optional[float], Optional[float]]:
        """Lấy lat/lon từ Wikipedia (parse HTML)."""
        wikipedia.set_lang(lang)
        try:
            search_results = wikipedia.search(place_name)
            if not search_results:
                return None, None

            # Tìm tiêu đề gần nhất
            page_title = max(
                search_results,
                key=lambda t: SequenceMatcher(None, place_name, t).ratio()
            )

            # Có thể ném DisambiguationError/HTTPError
            page = wikipedia.page(page_title, auto_suggest=False)

            # Parse HTML
            response = requests.get(page.url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            lat_span = soup.find("span", class_="latitude")
            lng_span = soup.find("span", class_="longitude")

            if lat_span and lng_span:
                lat = self.coord.convert_lat_lon(lat_span.text)
                lng = self.coord.convert_lat_lon(lng_span.text)
                # Nếu convert trả về str (không parse được) -> coi như thất bại
                if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
                    return float(lat), float(lng)

        except wikipedia.DisambiguationError as e:
            print(f"Wiki disambiguation: {e}")
        except wikipedia.PageError as e:
            print(f"Wiki page error: {e}")
        except requests.RequestException as e:
            print(f"HTTP error: {e}")
        except Exception as e:
            print(f"Wiki error: {e}")

        return None, None


class ProvinceNormalizer:
    """Chuẩn hóa tên tỉnh/thành phố và trích xuất địa chỉ."""

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

    @staticmethod
    def normalize_province(province_name: Optional[str]) -> Optional[str]:
        """Chuẩn hóa tên tỉnh/thành phố (trả về None nếu không hợp lệ)."""
        if not isinstance(province_name, str) or not province_name.strip():
            return None

        cleaned = province_name.lower().strip()
        cleaned = re.sub(r"\b(tp\.|thành phố|tỉnh)\b", "", cleaned, flags=re.IGNORECASE).strip()

        # Mapping đặc biệt
        if "phú quốc" in cleaned:
            return "Kiên Giang"
        if any(k in cleaned for k in ["hcm", "hồ chí minh", "ho chi minh"]):
            return "TP Hồ Chí Minh"
        if any(k in cleaned for k in ["hn", "hà nội", "ha noi"]):
            return "Hà Nội"
        if any(k in cleaned for k in ["đn", "da nang", "đà nẵng"]):
            return "Đà Nẵng"

        # Fuzzy match (so sánh lowercase)
        return max(
            ProvinceNormalizer.STANDARD_PROVINCES,
            key=lambda t: SequenceMatcher(None, cleaned, t.lower()).ratio()
        )

    @staticmethod
    def extract_address(address: str, vn_location: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Trích xuất (district, province, street) từ địa chỉ.
        vn_location chưa dùng, để mở rộng sau.
        """
        if not address:
            return None, None, None

        # Chuẩn hóa unicode + ký tự
        address = unicodedata.normalize("NFC", address)
        address = address.replace("Ð", "Đ")
        address = re.sub(r",?\s*(Việt Nam|VN)\b", "", address, flags=re.IGNORECASE).strip()

        # Lấy city/province
        city_match = re.search(r"(TP\.|Thành phố|Tỉnh)\s*([A-Za-zÀ-ỹ\s]+)", address)
        city = city_match.group(2).strip() if city_match else None
        province = ProvinceNormalizer.normalize_province(city) if city else None

        # Lấy district
        district_match = re.search(r"(Quận|Huyện|Thị xã)\s*([A-Za-zÀ-ỹ\s0-9]+)", address)
        district = district_match.group(0).strip() if district_match else None

        # street: có thể tách trước phần district/city (để TODO)
        street = None

        return district, province, street


class HotelFinder:
    """Tìm khách sạn gần các địa điểm du lịch hoặc theo tỉnh."""

    def __init__(self, neighbor_province: Dict[str, List[str]]):
        self.geolocator = Nominatim(user_agent="hotel_locator")
        self.neighbor_province = neighbor_province

    @staticmethod
    def is_similar(query: str, text: str, threshold: int = 80) -> bool:
        return fuzz.partial_ratio((query or "").lower(), (text or "").lower()) >= threshold

    def find_hotels_near_location(
        self,
        data_dict: Dict[str, Dict[str, Any]],
        places: List[str],
        location: str,
        max_distance_km: float = 20.0
    ) -> List[Dict[str, Any]]:
        """
        Trả về list khách sạn phù hợp:
        - Nếu không có 'places': lọc theo tỉnh/thành phố từ 'location'
        - Nếu có 'places': suy ra tỉnh liên quan từ geocode các place, rồi lọc
          các khách sạn cùng (hoặc lân cận) tỉnh và có mô tả match tên place.
        """
        places_location: List[Tuple[float, float]] = []
        relevant_provinces: set = set()

        # Trường hợp không có places: lọc theo tỉnh của location
        if not places:
            base_prov = ProvinceNormalizer.normalize_province(location)
            relevant_provinces = {base_prov} if base_prov else set()
            return [
                h for h in data_dict.values()
                if ProvinceNormalizer.normalize_province(h.get('province/ city', '')) in relevant_provinces
            ]

        # Có places: geocode từng place -> suy ra province + lân cận
        for place in places:
            try:
                loc = self.geolocator.geocode(place, timeout=10)
            except Exception:
                loc = None

            if loc:
                # Lấy province từ address geocoder trả về
                prov = ProvinceNormalizer.extract_address(loc.address, self.neighbor_province)[1]
                if prov:
                    # thêm tỉnh + các tỉnh lân cận
                    neighbors = self.neighbor_province.get(prov, [])
                    relevant_provinces.update(neighbors)
                    relevant_provinces.add(prov)
                places_location.append((float(loc.latitude), float(loc.longitude)))

        if not relevant_provinces:
            # fallback nhẹ: thử dùng location ban đầu
            base_prov = ProvinceNormalizer.normalize_province(location)
            if base_prov:
                relevant_provinces.add(base_prov)

        if not relevant_provinces:
            return []

        # Lọc khách sạn theo provinces
        relevant_hotels = [
            h for h in data_dict.values()
            if ProvinceNormalizer.normalize_province(h.get('province/ city', '')) in relevant_provinces
        ]

        # Nếu không có toạ độ khách sạn để tính khoảng cách, ưu tiên theo mô tả khớp tên place
        matched_hotels: List[Dict[str, Any]] = []
        for hotel in relevant_hotels:
            desc = (hotel.get('description') or "").lower()
            name = (hotel.get('hotel_name') or "").lower()
            combined = f"{name} {desc}"

            # Match theo tên địa danh người dùng đưa vào
            if any(self.is_similar(user_place, combined) for user_place in places):
                matched_hotels.append(hotel)

        # Nếu không match gì theo mô tả, trả lại toàn bộ theo province như fallback
        return matched_hotels if matched_hotels else relevant_hotels


class LocationScorer:
    """
    Tính điểm vị trí khách sạn dựa trên nhiều tiêu chí:
    - Độ phủ điểm đến user quan tâm
    - Tổng khoảng cách đến các điểm
    - Mật độ tiện ích xung quanh
    - Ưu tiên gần trung tâm (log scale)
    """

    @staticmethod
    def clean_distance(distance: Any) -> Optional[float]:
        """
        Chuẩn hoá về km (float). Hỗ trợ "850 m", "0.8 km", "1.2km", "khoảng 2 km", 1.2, 500 (m? -> không đoán).
        """
        if isinstance(distance, (int, float)):
            return float(distance)

        if distance is None:
            return None

        s = str(distance).strip().lower().replace(",", ".")
        # lấy số đầu tiên
        m = re.search(r"[-+]?\d*\.?\d+", s)
        if not m:
            return None
        try:
            num = float(m.group(0))
        except ValueError:
            return None

        # quyết định đơn vị
        if " m" in s or s.endswith("m"):
            return num / 1000.0
        # mặc định km nếu có "km" hoặc không rõ
        return num

    @staticmethod
    def is_similar(query: str, text: str, threshold: int = 80) -> bool:
        return fuzz.partial_ratio((query or "").lower(), (text or "").lower()) >= threshold

    @staticmethod
    def calculate_location_score(
        hotel: Dict[str, Any],
        user_places: List[str],
        province: Optional[str] = None,
        is_near_center: bool = False
    ) -> float:
        """
        Trả về điểm [0.0 - 1.0]:
        - Độ phủ điểm đến user quan tâm
        - Trung bình khoảng cách đến các điểm khớp
        - Mật độ tiện ích xung quanh
        - Ưu tiên gần trung tâm (log-scale)
        """
        nearby_places = hotel.get('nearby_places', []) or []

        # 1) Chuẩn hoá: {place -> distance_km}
        places_data: Dict[str, float] = {}
        for category in nearby_places:
            details = (category or {}).get('detail', {})
            if not isinstance(details, dict):
                continue
            for place, distance in details.items():
                dist_km = LocationScorer.clean_distance(distance)
                if dist_km is not None:
                    places_data[str(place)] = float(dist_km)

        # 2) Điểm độ phủ [0..1]
        matched_places: List[str] = []
        for user_place in (user_places or []):
            for place in places_data:
                if LocationScorer.is_similar(user_place, place):
                    matched_places.append(place)
                    break
        coverage_score = (len(matched_places) / len(user_places)) if user_places else 0.0

        # 3) Điểm khoảng cách [0..1] - trung bình khoảng cách (càng gần càng cao)
        distance_score = 0.0
        if matched_places:
            avg_distance = sum(places_data[p] for p in matched_places) / len(matched_places)
            max_distance = 20.0  # > 20km thì ~0 điểm
            distance_score = max(0.0, 1.0 - (avg_distance / max_distance))

        # 4) Điểm mật độ tiện ích [0..1] - log normalization
        amenities_count = len(places_data)
        amenities_score = min(1.0, math.log1p(amenities_count) / math.log1p(30.0))

        # 5) Điểm gần trung tâm [0..1]
        center_score = 0.0
        if is_near_center and province:
            # heuristic đơn giản
            center_keywords = ["trung tâm", "city center", "central", province.split()[0]]
            for place, dist in places_data.items():
                if any(k.lower() in place.lower() for k in center_keywords):
                    max_dist = 20.0
                    if dist <= max_dist:
                        center_score = max(0.0, 1.0 - math.log1p(dist) / math.log1p(max_dist))
                    break

        # 6) Tổng hợp
        total_score = (coverage_score + distance_score + amenities_score + center_score) / 4.0
        return round(total_score, 4)
