from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
from enum import Enum
import json
from datetime import datetime, date
import re


class RoomType(Enum):
    """Room type enumeration"""
    SINGLE = "Single Room"
    DOUBLE = "Double Room"
    TWIN = "Twin Room"
    TRIPLE = "Triple Room"
    FAMILY = "Family Room"
    SUITE = "Suite"
    DELUXE = "Deluxe Room"


class RoomLevel(Enum):
    """Room level enumeration"""
    STANDARD = "Standard"
    SUPERIOR = "Superior"
    DELUXE = "Deluxe"
    PREMIUM = "Premium"
    LUXURY = "Luxury"


class BedType(Enum):
    """Bed type enumeration"""
    SINGLE_BED = "giường đơn"
    DOUBLE_BED = "giường đôi"
    LARGE_DOUBLE_BED = "giường đôi lớn"
    TWIN_BEDS = "hai giường đơn"
    SOFA_BED = "giường sofa"


class GuestType(Enum):
    """Guest type enumeration"""
    COUPLE = "Cặp đôi"
    FAMILY = "Phòng gia đình"
    BUSINESS = "Công tác"
    SOLO = "Một mình"
    GROUP = "Nhóm bạn"


class ServiceCategory(Enum):
    """Service category enumeration"""
    TRANSPORT_PARKING = "Dịch vụ vận chuyển & bãi đỗ xe"
    EVENT_SUPPORT = "Dịch vụ sự kiện & hỗ trợ đặc biệt"
    ENTERTAINMENT_SPA = "Tiện ích giải trí & spa"
    FOOD_BEVERAGE = "Ăn uống"
    BUSINESS = "Dịch vụ kinh doanh"
    CLEANING = "Dịch vụ dọn dẹp"


class QualityCriteria(Enum):
    """Quality criteria enumeration"""
    VALUE = "Giá trị"
    CLEANLINESS = "Sạch sẽ"
    LOCATION_ENVIRONMENT = "Vị trí & môi trường"
    SERVICE_STAFF = "Dịch vụ & nhân viên"
    FACILITIES = "Tiện nghi"
    COMFORT = "Thoải mái"


@dataclass
class PriceRange:
    """Price range specification"""
    min_price: int
    max_price: int
    
    def __post_init__(self):
        if self.min_price < 0:
            raise ValueError("Minimum price cannot be negative")
        if self.max_price < self.min_price:
            raise ValueError("Maximum price cannot be less than minimum price")
    
    @classmethod
    def from_tuple(cls, price_tuple: Tuple[int, int]) -> 'PriceRange':
        """Create PriceRange from tuple"""
        return cls(min_price=price_tuple[0], max_price=price_tuple[1])
    
    def contains(self, price: int) -> bool:
        """Check if price is within range"""
        return self.min_price <= price <= self.max_price
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary"""
        return {"min_price": self.min_price, "max_price": self.max_price}


@dataclass
class HotelPolicies:
    """Hotel policies specification"""
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
        """Create HotelPolicies from dictionary"""
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
        """Convert to dictionary"""
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
    """
    Comprehensive hotel search query specification.
    
    This class defines all possible search criteria for hotel booking systems.
    """
    
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
        """Validate and process query after initialization"""
        self._validate_query()
        if self.query_id is None:
            self.query_id = self._generate_query_id()
    
    def _validate_query(self) -> None:
        """Validate query parameters"""
        if self.capacity <= 0:
            raise ValueError("Capacity must be positive")
        
        if self.stars_rating is not None and not (1 <= self.stars_rating <= 5):
            raise ValueError("Stars rating must be between 1 and 5")
        
        if self.distance_to_city_center is not None and self.distance_to_city_center < 0:
            raise ValueError("Distance to city center cannot be negative")
        
        if self.area is not None and self.area <= 0:
            raise ValueError("Room area must be positive")
    
    def _generate_query_id(self) -> str:
        """Generate unique query identifier"""
        timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
        location_hash = hash(f"{self.country}_{self.province}") % 10000
        return f"query_{timestamp}_{location_hash:04d}"
    
    @classmethod
    def from_dict(cls, query_dict: Dict[str, Any]) -> 'HotelQuery':
        """Create HotelQuery from dictionary"""
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
        """Convert query to dictionary"""
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
        """Convert query to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'HotelQuery':
        """Create HotelQuery from JSON string"""
        data = json.loads(json_str)
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls.from_dict(data)
    
    def matches_criteria(self, **filters) -> bool:
        """Check if query matches given filters"""
        for key, value in filters.items():
            if not hasattr(self, key):
                continue
            
            query_value = getattr(self, key)
            
            if isinstance(query_value, list):
                if not any(item in query_value for item in (value if isinstance(value, list) else [value])):
                    return False
            elif query_value != value:
                return False
        
        return True
    
    def get_location_summary(self) -> str:
        """Get human-readable location summary"""
        parts = [self.province] if self.province else []
        if self.nearby_places:
            parts.append(f"gần {', '.join(self.nearby_places[:2])}")
        if self.is_near_center:
            parts.append("trung tâm thành phố")
        return ", ".join(parts) if parts else self.country
    
    def get_price_summary(self) -> str:
        """Get human-readable price summary"""
        if not self.price_range:
            return "Không giới hạn giá"
        
        min_price = f"{self.price_range.min_price:,}đ"
        max_price = f"{self.price_range.max_price:,}đ"
        return f"{min_price} - {max_price}"
    
    def get_room_summary(self) -> str:
        """Get human-readable room summary"""
        parts = []
        
        if self.room_type:
            parts.append(self.room_type)
        
        if self.capacity:
            parts.append(f"{self.capacity} người")
        
        if self.bed_type:
            parts.append(self.bed_type)
        
        if self.area:
            parts.append(f"{self.area}m²")
        
        return ", ".join(parts) if parts else "Không xác định"


class HotelQueryBuilder:
    """Builder pattern for creating HotelQuery objects"""
    
    def __init__(self):
        self._query_data = {}
    
    def location(self, province: str, nearby_places: List[str] = None, 
                near_center: bool = True, distance_to_center: int = None) -> 'HotelQueryBuilder':
        """Set location criteria"""
        self._query_data.update({
            'province': province,
            'nearby_places': nearby_places or [],
            'is_near_center': near_center,
            'distance_to_city_center': distance_to_center
        })
        return self
    
    def price_range(self, min_price: int, max_price: int) -> 'HotelQueryBuilder':
        """Set price range"""
        self._query_data['price_range'] = (min_price, max_price)
        return self
    
    def hotel_features(self, stars: int = None, services: List[str] = None,
                      amenities: List[str] = None, criteria: List[str] = None) -> 'HotelQueryBuilder':
        """Set hotel features"""
        if stars is not None:
            self._query_data['stars_rating'] = stars
        if services:
            self._query_data['services'] = services
        if amenities:
            self._query_data['amenities'] = amenities
        if criteria:
            self._query_data['criteria'] = criteria
        return self
    
    def room_specifications(self, capacity: int = None, room_type: str = None,
                          bed_type: str = None, area: int = None,
                          room_view: List[str] = None) -> 'HotelQueryBuilder':
        """Set room specifications"""
        if capacity is not None:
            self._query_data['capacity'] = capacity
        if room_type:
            self._query_data['room_type'] = room_type
        if bed_type:
            self._query_data['bed_type'] = bed_type
        if area:
            self._query_data['area'] = area
        if room_view:
            self._query_data['room_view'] = room_view
        return self
    
    def booking_preferences(self, flexibility: List[str] = None,
                          breakfast: bool = None) -> 'HotelQueryBuilder':
        """Set booking preferences"""
        if flexibility:
            self._query_data['booking_flexibility'] = flexibility
        if breakfast is not None:
            self._query_data['included_breakfast'] = breakfast
        return self
    
    def build(self) -> HotelQuery:
        """Build the final HotelQuery object"""
        return HotelQuery.from_dict(self._query_data)


class HotelQueryManager:
    """Manager class for handling multiple hotel queries"""
    
    def __init__(self):
        self.queries: Dict[str, HotelQuery] = {}
    
    def add_query(self, query: HotelQuery) -> str:
        """Add a query and return its ID"""
        self.queries[query.query_id] = query
        return query.query_id
    
    def get_query(self, query_id: str) -> Optional[HotelQuery]:
        """Get query by ID"""
        return self.queries.get(query_id)
    
    def remove_query(self, query_id: str) -> bool:
        """Remove query by ID"""
        return self.queries.pop(query_id, None) is not None
    
    def find_queries(self, **filters) -> List[HotelQuery]:
        """Find queries matching filters"""
        return [
            query for query in self.queries.values()
            if query.matches_criteria(**filters)
        ]
    
    def get_queries_by_location(self, province: str) -> List[HotelQuery]:
        """Get all queries for a specific province"""
        return [
            query for query in self.queries.values()
            if query.province == province
        ]
    
    def export_queries(self, filename: str) -> None:
        """Export all queries to JSON file"""
        data = {
            query_id: query.to_dict()
            for query_id, query in self.queries.items()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def import_queries(self, filename: str) -> int:
        """Import queries from JSON file"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for query_data in data.values():
            try:
                query = HotelQuery.from_dict(query_data)
                self.queries[query.query_id] = query
                count += 1
            except Exception as e:
                print(f"Error importing query: {e}")
        
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored queries"""
        if not self.queries:
            return {}
        
        provinces = [q.province for q in self.queries.values() if q.province]
        price_ranges = [q.price_range for q in self.queries.values() if q.price_range]
        capacities = [q.capacity for q in self.queries.values()]
        
        return {
            'total_queries': len(self.queries),
            'provinces': list(set(provinces)),
            'most_common_province': max(set(provinces), key=provinces.count) if provinces else None,
            'average_capacity': sum(capacities) / len(capacities) if capacities else 0,
            'average_min_price': sum(pr.min_price for pr in price_ranges) / len(price_ranges) if price_ranges else 0,
            'average_max_price': sum(pr.max_price for pr in price_ranges) / len(price_ranges) if price_ranges else 0
        }


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Create query from dictionary (like your original data)
    sample_query_data = {
        'province': 'Lâm Đồng',
        'nearby_places': ['Chợ Đà Lạt', 'Hồ Xuân Hương', 'Quảng trường Lâm Viên', 'Sân bay Liên Khương'],
        'is_near_center': True,
        'policies_hotels': {
            'Nhận phòng': 'Từ 14:00 - 00:00',
            'Trả phòng': 'Từ 00:00 - 12:00',
            'Hủy đặt phòng/ Trả trước': 'Các chính sách hủy và thanh toán trước sẽ khác nhau tùy vào từng loại chỗ nghỉ.',
            'Trẻ em và giường': 'Phù hợp cho tất cả trẻ em.',
            'Không giới hạn độ tuổi': 'Không có yêu cầu về độ tuổi khi nhận phòng',
            'Vật nuôi': 'Vật nuôi không được phép.',
            'Giờ giới nghiêm': 'Cổng vào chỗ nghỉ sẽ đóng trong khoảng từ 22:00 - 03:30',
            'Chỉ thanh toán bằng tiền mặt': 'Chỗ nghỉ này chỉ chấp nhận thanh toán bằng tiền mặt.'
        },
        'price_range': (300000, 800000),
        'stars_rating': 3,
        'services': ['Dịch vụ vận chuyển & bãi đỗ xe'],
        'criteria': ['Giá trị', 'Sạch sẽ'],
        'amenities': ['TV', 'máy sưởi'],
        'booking_flexibility': ['hủy miễn phí', 'thanh toán khi nhận phòng'],
        'distance_to_city_center': 3,
        'capacity': 2,
        'country': 'Việt Nam',
        'state': 'Phòng gia đình',
        'room_type': 'Double Room',
        'included_breakfast': False,
        'room_level': 'Standard',
        'bed_type': 'giường đơn',
        'area': 25,
        'room_view': ['Nhìn ra thành phố']
    }
    
    # Create query from dictionary
    query1 = HotelQuery.from_dict(sample_query_data)
    print("Query 1 ID:", query1.query_id)
    print("Location:", query1.get_location_summary())
    print("Price:", query1.get_price_summary())
    print("Room:", query1.get_room_summary())
    print()
    
    # Example 2: Use builder pattern
    query2 = (HotelQueryBuilder()
              .location("Thành phố Hồ Chí Minh", ["Chợ Bến Thành", "Phố đi bộ Nguyễn Huệ"], True, 5)
              .price_range(500000, 2800000)
              .hotel_features(stars=4, services=["Dịch vụ sự kiện & hỗ trợ đặc biệt"], 
                            amenities=["TV màn hình phẳng", "máy lạnh"])
              .room_specifications(capacity=2, room_type="Double Room", bed_type="giường đôi lớn", area=35)
              .booking_preferences(flexibility=["hủy miễn phí"], breakfast=True)
              .build())
    
    print("Query 2 ID:", query2.query_id)
    print("Location:", query2.get_location_summary())
    print("Price:", query2.get_price_summary())
    print()
    
    # Example 3: Query manager
    manager = HotelQueryManager()
    manager.add_query(query1)
    manager.add_query(query2)
    
    print("Manager Statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Find queries by criteria
    dalat_queries = manager.find_queries(province="Lâm Đồng")
    print(f"\nFound {len(dalat_queries)} queries for Đà Lạt")
    
    # Export/Import example
    manager.export_queries("sample_queries.json")
    print("Queries exported to sample_queries.json")