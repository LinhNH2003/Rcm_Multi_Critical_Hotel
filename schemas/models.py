"""
Data schemas and models for the hotel recommendation system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any, Tuple
from datetime import datetime, date
from enum import Enum
import json


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


@dataclass
class PriceRange:
    """Price range specification."""
    min_price: int
    max_price: int
    
    def __post_init__(self):
        if self.min_price < 0:
            raise ValueError("Minimum price cannot be negative")
        if self.max_price < self.min_price:
            raise ValueError("Maximum price cannot be less than minimum price")
    
    @classmethod
    def from_tuple(cls, price_tuple: Tuple[int, int]) -> 'PriceRange':
        """Create PriceRange from tuple."""
        return cls(min_price=price_tuple[0], max_price=price_tuple[1])
    
    def contains(self, price: int) -> bool:
        """Check if price is within range."""
        return self.min_price <= price <= self.max_price
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {"min_price": self.min_price, "max_price": self.max_price}


@dataclass
class HotelPolicies:
    """Hotel policies specification."""
    check_in: str = "Từ 14:00 - 00:00"
    check_out: str = "Từ 00:00 - 12:00"
    cancellation_policy: str = ""
    children_policy: str = ""
    age_restriction: Optional[str] = None
    pet_policy: str = "Vật nuôi không được phép."
    curfew: Optional[str] = None
    cash_only: bool = False
    smoking_policy: Optional[str] = None
    party_policy: Optional[str] = None
    
    @classmethod
    def from_dict(cls, policies_dict: Dict[str, str]) -> 'HotelPolicies':
        """Create HotelPolicies from dictionary."""
        return cls(
            check_in=policies_dict.get('Nhận phòng', ''),
            check_out=policies_dict.get('Trả phòng', ''),
            cancellation_policy=policies_dict.get('Hủy đặt phòng/ Trả trước', ''),
            children_policy=policies_dict.get('Trẻ em và giường', ''),
            age_restriction=policies_dict.get('Không giới hạn độ tuổi'),
            pet_policy=policies_dict.get('Vật nuôi', 'Vật nuôi không được phép.'),
            curfew=policies_dict.get('Giờ giới nghiêm'),
            cash_only=bool(policies_dict.get('Chỉ thanh toán bằng tiền mặt')),
            smoking_policy=policies_dict.get('Hút thuốc'),
            party_policy=policies_dict.get('Tiệc tùng')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'Nhận phòng': self.check_in,
            'Trả phòng': self.check_out,
            'Hủy đặt phòng/ Trả trước': self.cancellation_policy,
            'Trẻ em và giường': self.children_policy,
            'Không giới hạn độ tuổi': self.age_restriction,
            'Vật nuôi': self.pet_policy,
            'Giờ giới nghiêm': self.curfew,
            'Chỉ thanh toán bằng tiền mặt': self.cash_only,
            'Hút thuốc': self.smoking_policy,
            'Tiệc tùng': self.party_policy
        }


@dataclass
class HotelQuery:
    """Comprehensive hotel search query specification."""
    
    # Location criteria
    country: str = "Việt Nam"
    province: str = ""
    nearby_places: List[str] = field(default_factory=list)
    is_near_center: bool = True
    distance_to_city_center: Optional[int] = None  # in km
    
    # Pricing
    price_range: Optional[PriceRange] = None
    
    # Hotel characteristics
    stars_rating: Optional[int] = None
    services: List[str] = field(default_factory=list)
    criteria: List[str] = field(default_factory=list)
    amenities: List[str] = field(default_factory=list)
    
    # Booking preferences
    booking_flexibility: List[str] = field(default_factory=list)
    included_breakfast: Optional[bool] = None
    
    # Room specifications
    capacity: int = 2
    state: Optional[str] = None  # Guest type
    room_type: Optional[str] = None
    room_level: Optional[str] = None
    bed_type: Optional[str] = None
    area: Optional[int] = None  # in square meters
    room_view: List[str] = field(default_factory=list)
    
    # Policies
    policies_hotels: Optional[HotelPolicies] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    query_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate and process query after initialization."""
        self._validate_query()
        if self.query_id is None:
            self.query_id = self._generate_query_id()
    
    def _validate_query(self) -> None:
        """Validate query parameters."""
        if self.capacity <= 0:
            raise ValueError("Capacity must be positive")
        
        if self.stars_rating is not None and not (1 <= self.stars_rating <= 5):
            raise ValueError("Stars rating must be between 1 and 5")
        
        if self.distance_to_city_center is not None and self.distance_to_city_center < 0:
            raise ValueError("Distance to city center cannot be negative")
        
        if self.area is not None and self.area <= 0:
            raise ValueError("Room area must be positive")
    
    def _generate_query_id(self) -> str:
        """Generate unique query identifier."""
        timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
        location_hash = hash(f"{self.country}_{self.province}") % 10000
        return f"query_{timestamp}_{location_hash:04d}"
    
    @classmethod
    def from_dict(cls, query_dict: Dict[str, Any]) -> 'HotelQuery':
        """Create HotelQuery from dictionary."""
        # Handle price range
        price_range = None
        if 'price_range' in query_dict and query_dict['price_range']:
            price_range = PriceRange.from_tuple(query_dict['price_range'])
        
        # Handle policies
        policies = None
        if 'policies_hotels' in query_dict and query_dict['policies_hotels']:
            policies = HotelPolicies.from_dict(query_dict['policies_hotels'])
        
        # Create query
        query = cls(
            country=query_dict.get('country', 'Việt Nam'),
            province=query_dict.get('province', ''),
            nearby_places=query_dict.get('nearby_places', []),
            is_near_center=query_dict.get('is_near_center', True),
            distance_to_city_center=query_dict.get('distance_to_city_center'),
            price_range=price_range,
            stars_rating=query_dict.get('stars_rating'),
            services=query_dict.get('services', []),
            criteria=query_dict.get('criteria', []),
            amenities=query_dict.get('amenities', []),
            booking_flexibility=query_dict.get('booking_flexibility', []),
            included_breakfast=query_dict.get('included_breakfast'),
            capacity=query_dict.get('capacity', 2),
            state=query_dict.get('state'),
            room_type=query_dict.get('room_type'),
            room_level=query_dict.get('room_level'),
            bed_type=query_dict.get('bed_type'),
            area=query_dict.get('area'),
            room_view=query_dict.get('room_view', []),
            policies_hotels=policies
        )
        
        return query
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary."""
        result = {
            'query_id': self.query_id,
            'country': self.country,
            'province': self.province,
            'nearby_places': self.nearby_places,
            'is_near_center': self.is_near_center,
            'distance_to_city_center': self.distance_to_city_center,
            'price_range': self.price_range.to_dict() if self.price_range else None,
            'stars_rating': self.stars_rating,
            'services': self.services,
            'criteria': self.criteria,
            'amenities': self.amenities,
            'booking_flexibility': self.booking_flexibility,
            'included_breakfast': self.included_breakfast,
            'capacity': self.capacity,
            'state': self.state,
            'room_type': self.room_type,
            'room_level': self.room_level,
            'bed_type': self.bed_type,
            'area': self.area,
            'room_view': self.room_view,
            'policies_hotels': self.policies_hotels.to_dict() if self.policies_hotels else None,
            'created_at': self.created_at.isoformat()
        }
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert query to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'HotelQuery':
        """Create HotelQuery from JSON string."""
        data = json.loads(json_str)
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls.from_dict(data)


@dataclass
class HotelInfo:
    """Hotel information schema."""
    id: str
    title: str
    url: str
    lat_lng: Tuple[float, float]
    rate: Optional[float] = None
    sub_rate: Optional[Dict[str, str]] = None
    review_count: Optional[str] = None
    address: Optional[str] = None
    popular_facilities: List[str] = field(default_factory=list)
    description: Optional[str] = None
    hotel_facilities: Optional[List[Dict[str, Any]]] = None
    images: List[Dict[str, str]] = field(default_factory=list)
    nearby_places: Optional[List[Dict[str, Any]]] = None
    policies_hotels: Optional[Dict[str, str]] = None
    info_roomtypes: Optional[Dict[str, Any]] = None
    stars_rating: Optional[int] = None
    time_crawl: Optional[str] = None


@dataclass
class RoomInfo:
    """Room information schema."""
    id: str
    room_id: str
    name: str
    original_price: Optional[str] = None
    current_price: Optional[str] = None
    capacity: Optional[str] = None
    images: List[Dict[str, str]] = field(default_factory=list)
    amenities: List[str] = field(default_factory=list)
    bed_types: Optional[str] = None
    facilities: Optional[Dict[str, List[str]]] = None
    room_service_included: Optional[str] = None


@dataclass
class ReviewInfo:
    """Review information schema."""
    name: Optional[str] = None
    country: Optional[str] = None
    room: Optional[str] = None
    date: Optional[str] = None
    state: Optional[str] = None
    review_title: Optional[str] = None
    review_date: Optional[str] = None
    review_score: Optional[str] = None
    review_positive: Optional[str] = None
    review_negative: Optional[str] = None
    review_photo: Optional[Dict[str, str]] = None


@dataclass
class HotelData:
    """Complete hotel data schema."""
    id: str
    stars_rating: Optional[int] = None
    info: Optional[HotelInfo] = None
    reviews: List[ReviewInfo] = field(default_factory=list)
    time_crawl: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'HotelData':
        """Create HotelData from dictionary."""
        hotel_info = None
        if 'info' in data_dict and data_dict['info']:
            hotel_info = HotelInfo(**data_dict['info'])
        
        reviews = []
        if 'reviews' in data_dict and data_dict['reviews']:
            reviews = [ReviewInfo(**review) for review in data_dict['reviews']]
        
        return cls(
            id=data_dict.get('id', ''),
            stars_rating=data_dict.get('stars_rating'),
            info=hotel_info,
            reviews=reviews,
            time_crawl=data_dict.get('time_crawl')
        )


@dataclass
class RecommendationResult:
    """Recommendation result schema."""
    hotel_id: str
    similarity_score: float
    original_similarity: float
    amenities: List[str] = field(default_factory=list)
    num_amenities: int = 0
    matched_amenities: List[str] = field(default_factory=list)
    num_matches: int = 0
    room_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hotel_id': self.hotel_id,
            'similarity_score': self.similarity_score,
            'original_similarity': self.original_similarity,
            'amenities': self.amenities,
            'num_amenities': self.num_amenities,
            'matched_amenities': self.matched_amenities,
            'num_matches': self.num_matches,
            'room_id': self.room_id
        }


@dataclass
class ScoringResult:
    """Scoring result schema."""
    hotel_id: str
    score: float
    score_type: str  # e.g., 'amenity', 'location', 'price', 'rating', 'policy'
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hotel_id': self.hotel_id,
            'score': self.score,
            'score_type': self.score_type,
            'details': self.details or {}
        }


@dataclass
class FinalRecommendation:
    """Final recommendation result schema."""
    hotel_id: str
    final_score: float
    individual_scores: Dict[str, float] = field(default_factory=dict)
    hotel_info: Optional[HotelInfo] = None
    recommendation_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hotel_id': self.hotel_id,
            'final_score': self.final_score,
            'individual_scores': self.individual_scores,
            'hotel_info': self.hotel_info.__dict__ if self.hotel_info else None,
            'recommendation_reason': self.recommendation_reason
        }


# Validation schemas using Pydantic-like structure
class ValidationError(Exception):
    """Custom validation error."""
    pass


def validate_hotel_query(query: HotelQuery) -> bool:
    """Validate hotel query."""
    if query.capacity <= 0:
        raise ValidationError("Capacity must be positive")
    
    if query.stars_rating is not None and not (1 <= query.stars_rating <= 5):
        raise ValidationError("Stars rating must be between 1 and 5")
    
    if query.distance_to_city_center is not None and query.distance_to_city_center < 0:
        raise ValidationError("Distance to city center cannot be negative")
    
    if query.area is not None and query.area <= 0:
        raise ValidationError("Room area must be positive")
    
    return True


def validate_price_range(price_range: PriceRange) -> bool:
    """Validate price range."""
    if price_range.min_price < 0:
        raise ValidationError("Minimum price cannot be negative")
    
    if price_range.max_price < price_range.min_price:
        raise ValidationError("Maximum price cannot be less than minimum price")
    
    return True

