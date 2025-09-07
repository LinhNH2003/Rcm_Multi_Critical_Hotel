"""
Constants and enumerations for the hotel recommendation system.
"""

from enum import Enum
from typing import Dict, List, Tuple


class RoomType(Enum):
    """Room type enumeration."""
    SINGLE = "Single Room"
    DOUBLE = "Double Room"
    TWIN = "Twin Room"
    TRIPLE = "Triple Room"
    FAMILY = "Family Room"
    SUITE = "Suite"
    DELUXE = "Deluxe Room"
    BUNGALOW = "Bungalow"
    VILLA = "Villa"
    APARTMENT = "Apartment/Studio/Penthouse"


class RoomLevel(Enum):
    """Room level enumeration."""
    STANDARD = "Standard"
    SUPERIOR = "Superior"
    DELUXE = "Deluxe"
    PREMIUM = "Premium"
    LUXURY = "Luxury"
    EXECUTIVE = "Executive"
    SIGNATURE = "Signature"
    VIP = "VIP"


class BedType(Enum):
    """Bed type enumeration."""
    SINGLE_BED = "giường đơn"
    DOUBLE_BED = "giường đôi"
    LARGE_DOUBLE_BED = "giường đôi lớn"
    EXTRA_LARGE_DOUBLE_BED = "giường đôi cực lớn"
    TWIN_BEDS = "hai giường đơn"
    SOFA_BED = "giường sofa"
    BUNK_BED = "giường tầng"
    FUTON_BED = "giường nệm futon kiểu Nhật"


class GuestType(Enum):
    """Guest type enumeration."""
    COUPLE = "Cặp đôi"
    FAMILY = "Phòng gia đình"
    BUSINESS = "Công tác"
    SOLO = "Một mình"
    GROUP = "Nhóm bạn"


class ServiceCategory(Enum):
    """Service category enumeration."""
    ROOM_AMENITIES = "Tiện ích phòng nghỉ"
    TECH_WORK = "Tiện ích công nghệ & làm việc"
    FOOD_BEVERAGE = "Dịch vụ ẩm thực"
    ENTERTAINMENT_SPA = "Tiện ích giải trí & spa"
    CHILDREN_AMENITIES = "Tiện ích dành cho trẻ em"
    GENERAL_SERVICES = "Dịch vụ & cơ sở vật chất chung"
    SECURITY_SAFETY = "Dịch vụ bảo mật & an toàn"
    EVENT_SUPPORT = "Dịch vụ sự kiện & hỗ trợ đặc biệt"
    TRANSPORT_PARKING = "Dịch vụ vận chuyển & bãi đỗ xe"


class QualityCriteria(Enum):
    """Quality criteria enumeration."""
    LOCATION_ENVIRONMENT = "Vị trí & môi trường"
    ROOM_AMENITIES = "Phòng nghỉ & tiện nghi"
    CLEANLINESS = "Sạch sẽ"
    SERVICE_STAFF = "Dịch vụ & nhân viên"
    WIFI = "Wi-Fi"
    FOOD = "Ẩm thực"
    FACILITIES = "Cơ sở vật chất"
    VALUE = "Giá trị"
    SECURITY = "An ninh"


class ScoringMethod(Enum):
    """Scoring method enumeration."""
    EQUAL_WEIGHTS = "equal_weights"
    WEIGHTED_RATIO = "weighted_ratio"
    REVIEW_COUNT_BASED = "review_count_based"


class ConversionMode(Enum):
    """Conversion mode enumeration."""
    SHORTEN = "shorten"
    EXPAND = "expand"


class Category(Enum):
    """Category enumeration."""
    SERVICES = "services"
    CRITERIA = "criteria"


# Service mappings
SERVICE_ID_TO_GROUP: Dict[int, str] = {
    1: ServiceCategory.ROOM_AMENITIES.value,
    3: ServiceCategory.ROOM_AMENITIES.value,
    14: ServiceCategory.ROOM_AMENITIES.value,
    2: ServiceCategory.TECH_WORK.value,
    18: ServiceCategory.TECH_WORK.value,
    4: ServiceCategory.FOOD_BEVERAGE.value,
    13: ServiceCategory.ENTERTAINMENT_SPA.value,
    5: ServiceCategory.ENTERTAINMENT_SPA.value,
    11: ServiceCategory.ENTERTAINMENT_SPA.value,
    16: ServiceCategory.ENTERTAINMENT_SPA.value,
    6: ServiceCategory.CHILDREN_AMENITIES.value,
    7: ServiceCategory.GENERAL_SERVICES.value,
    15: ServiceCategory.GENERAL_SERVICES.value,
    8: ServiceCategory.SECURITY_SAFETY.value,
    9: ServiceCategory.EVENT_SUPPORT.value,
    10: ServiceCategory.EVENT_SUPPORT.value,
    12: ServiceCategory.TRANSPORT_PARKING.value,
    17: ServiceCategory.TRANSPORT_PARKING.value
}

SERVICE_TEXT_TO_GROUP: Dict[str, str] = {
    "Tiện ích phòng": ServiceCategory.ROOM_AMENITIES.value,
    "Tiện ích công nghệ & kết nối": ServiceCategory.TECH_WORK.value,
    "Tiện ích phòng tắm": ServiceCategory.ROOM_AMENITIES.value,
    "Dịch vụ ẩm thực": ServiceCategory.FOOD_BEVERAGE.value,
    "Tiện ích thư giãn & giải trí": ServiceCategory.ENTERTAINMENT_SPA.value,
    "Tiện ích dành cho trẻ em": ServiceCategory.CHILDREN_AMENITIES.value,
    "Dịch vụ khách sạn": ServiceCategory.GENERAL_SERVICES.value,
    "Dịch vụ bảo mật & an toàn": ServiceCategory.SECURITY_SAFETY.value,
    "Dịch vụ tổ chức sự kiện": ServiceCategory.EVENT_SUPPORT.value,
    "Tiện nghi hỗ trợ đặc biệt": ServiceCategory.EVENT_SUPPORT.value,
    "Dịch vụ giải trí ngoài trời": ServiceCategory.ENTERTAINMENT_SPA.value,
    "Dịch vụ bãi đỗ xe": ServiceCategory.TRANSPORT_PARKING.value,
    "Tiện ích thiên nhiên": ServiceCategory.ENTERTAINMENT_SPA.value,
    "Hệ thống điều hòa & sưởi ấm": ServiceCategory.ROOM_AMENITIES.value,
    "Cơ sở vật chất chung": ServiceCategory.GENERAL_SERVICES.value,
    "Dịch vụ chăm sóc sức khỏe & spa": ServiceCategory.ENTERTAINMENT_SPA.value,
    "Dịch vụ đưa đón & phương tiện di chuyển": ServiceCategory.TRANSPORT_PARKING.value,
    "Tiện ích công việc": ServiceCategory.TECH_WORK.value
}

SERVICE_GROUP_TO_TEXTS: Dict[str, List[str]] = {
    ServiceCategory.ROOM_AMENITIES.value: ["Tiện ích phòng", "Tiện ích phòng tắm", "Hệ thống điều hòa & sưởi ấm"],
    ServiceCategory.TECH_WORK.value: ["Tiện ích công nghệ & kết nối", "Tiện ích công việc"],
    ServiceCategory.FOOD_BEVERAGE.value: ["Dịch vụ ẩm thực"],
    ServiceCategory.ENTERTAINMENT_SPA.value: ["Tiện ích thư giãn & giải trí", "Dịch vụ giải trí ngoài trời", 
                                            "Dịch vụ chăm sóc sức khỏe & spa"],
    ServiceCategory.CHILDREN_AMENITIES.value: ["Tiện ích dành cho trẻ em"],
    ServiceCategory.GENERAL_SERVICES.value: ["Dịch vụ khách sạn", "Cơ sở vật chất chung"],
    ServiceCategory.SECURITY_SAFETY.value: ["Dịch vụ bảo mật & an toàn"],
    ServiceCategory.EVENT_SUPPORT.value: ["Dịch vụ tổ chức sự kiện", "Tiện nghi hỗ trợ đặc biệt"],
    ServiceCategory.TRANSPORT_PARKING.value: ["Dịch vụ bãi đỗ xe", "Dịch vụ đưa đón & phương tiện di chuyển"]
}

# Criteria mappings
CRITERIA_ID_TO_GROUP: Dict[int, str] = {
    1: QualityCriteria.LOCATION_ENVIRONMENT.value,
    14: QualityCriteria.LOCATION_ENVIRONMENT.value,
    15: QualityCriteria.LOCATION_ENVIRONMENT.value,
    2: QualityCriteria.ROOM_AMENITIES.value,
    4: QualityCriteria.ROOM_AMENITIES.value,
    9: QualityCriteria.ROOM_AMENITIES.value,
    13: QualityCriteria.ROOM_AMENITIES.value,
    3: QualityCriteria.CLEANLINESS.value,
    5: QualityCriteria.SERVICE_STAFF.value,
    6: QualityCriteria.SERVICE_STAFF.value,
    7: QualityCriteria.WIFI.value,
    8: QualityCriteria.FOOD.value,
    10: QualityCriteria.FACILITIES.value,
    16: QualityCriteria.FACILITIES.value,
    11: QualityCriteria.VALUE.value,
    12: QualityCriteria.SECURITY.value,
}

CRITERIA_TEXT_TO_GROUP: Dict[str, str] = {
    "location": QualityCriteria.LOCATION_ENVIRONMENT.value,
    "room": QualityCriteria.ROOM_AMENITIES.value,
    "cleanliness": QualityCriteria.CLEANLINESS.value,
    "comfort": QualityCriteria.ROOM_AMENITIES.value,
    "service": QualityCriteria.SERVICE_STAFF.value,
    "staff": QualityCriteria.SERVICE_STAFF.value,
    "wifi": QualityCriteria.WIFI.value,
    "food": QualityCriteria.FOOD.value,
    "bathroom": QualityCriteria.ROOM_AMENITIES.value,
    "parking": QualityCriteria.FACILITIES.value,
    "value": QualityCriteria.VALUE.value,
    "facilities": QualityCriteria.FACILITIES.value,
    "air conditioning": QualityCriteria.ROOM_AMENITIES.value,
    "view": QualityCriteria.LOCATION_ENVIRONMENT.value,
    "environment": QualityCriteria.LOCATION_ENVIRONMENT.value,
    "security": QualityCriteria.SECURITY.value
}

CRITERIA_GROUP_TO_TEXTS: Dict[str, List[str]] = {
    QualityCriteria.LOCATION_ENVIRONMENT.value: ["location", "view", "environment"],
    QualityCriteria.ROOM_AMENITIES.value: ["room", "comfort", "bathroom", "air conditioning"],
    QualityCriteria.CLEANLINESS.value: ["cleanliness"],
    QualityCriteria.SERVICE_STAFF.value: ["service", "staff"],
    QualityCriteria.WIFI.value: ["wifi"],
    QualityCriteria.FOOD.value: ["food"],
    QualityCriteria.FACILITIES.value: ["facilities", "parking"],
    QualityCriteria.SECURITY.value: ["security"],
    QualityCriteria.VALUE.value: ["value"]
}

# Amenities mapping
AMENITIES: Dict[int, str] = {
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

# Room type lists
ROOM_TYPE_LIST: List[str] = [
    'Bungalow', 'Single Room', 'Suite', 'Family Room', 'Group Room',
    'Small Family Room', 'Double Room', 'Deluxe', 'Apartment/Studio/Penthouse',
    'Villa / Bungalow', 'Mansion/Residence', 'Family/Group Room',
    'Other', 'Chalet', 'Villa', 'Twin Room ', 'Cabin', 'Dormitory'
]

ROOM_LEVEL_LIST: List[str] = [
    'Standard', 'Junior', 'Royal', 'Premium', 'Premier', 
    'Luxury', 'Executive', 'Signature', 'Senior', 'VIP'
]

BED_TYPE_LIST: List[str] = [
    "giường đôi cực lớn", "giường đôi lớn", "giường đôi",
    "giường đơn", "giường sofa", "giường tầng", "giường nệm futon kiểu Nhật"
]

# Hotel criteria constants
ALL_CRITERIA: List[str] = [
    'location', 'room', 'cleanliness', 'comfort', 'service',
    'staff', 'wifi', 'food', 'bathroom', 'parking',
    'value', 'facilities', 'air conditioning', 'view',
    'environment', 'security'
]

# Vietnamese provinces
STANDARD_PROVINCES: List[str] = [
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

# Scoring constants
DEFAULT_WEIGHTS: Dict[str, float] = {
    'avg_score': 0.2,
    'count_review': 0.6,
    'image_ratio': 0.1,
    'long_review_ratio': 0.1
}

DYNAMIC_WEIGHTS_HIGH: Dict[str, float] = {
    'avg_score': 0.3,
    'count_review': 0.5,
    'image_ratio': 0.1,
    'long_review_ratio': 0.1
}

# Review quality constants
LONG_REVIEW_THRESHOLD: int = 150
IMAGE_RATIO_THRESHOLD: float = 0.5
LONG_REVIEW_RATIO_THRESHOLD: float = 0.5

# Policy patterns
POLICY_PATTERNS: Dict[str, str] = {
    "thanh_toan_truoc": r"\nThanh toán trước[^\n]*",
    "khong_can_thanh_toan_truoc": r"Không cần thanh toán trước - thanh toán tại chỗ nghỉ",
    "huy_mien_phi": r"(Hủy miễn phí trước \d{1,2} tháng \d{1,2}, \d{4})|(Miễn phí hủy đến.*? \d{1,2} tháng \d{1,2}, \d{4})",
    "khong_can_the_tin_dung": r"Không cần thẻ tín dụng",
    "linh_dong_doi_ngay": r"Linh động đổi ngày khi kế hoạch thay đổi",
    "khong_hoan_tien": r"Không hoàn tiền",
    "bao_gom_bua_sang": r"(Bao bữa sáng|Bao gồm bữa sáng|Bữa sáng [^\n]*|Giá bao gồm bữa sáng)"
}

# Sub-rating key mapping
SUB_RATING_KEY_MAP: Dict[str, str] = {
    "staff": "Nhân viên phục vụ",
    "facilities": "Tiện nghi",
    "cleanliness": "Sạch sẽ",
    "comfort": "Thoải mái",
    "value": "Đáng giá tiền",
    "location": "Địa điểm",
    "wifi": "WiFi miễn phí"
}

# Default configuration values
DEFAULT_CONFIG = {
    "bayesian_constant": 100,
    "default_weight": 1/16,
    "quality_weight": 0.8,
    "default_quality": 0.5,
    "global_score": 0.5,
    "quality_threshold": 0.0,
    "similarity_threshold": 0.90,
    "normalization_factor_base": 10.0,
    "beta_left": 2,
    "beta_right": 10,
    "max_distance_km": 20.0,
    "fuzzy_threshold": 80
}

