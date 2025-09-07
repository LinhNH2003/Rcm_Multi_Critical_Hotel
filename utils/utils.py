"""
Hotel recommendation system utilities for handling services, criteria, and scoring.
"""

from typing import Dict, List, Tuple, Union, Any, Optional
from collections import defaultdict
from enum import Enum


class Category(Enum):
    """Categories for conversion operations."""
    SERVICES = "services"
    CRITERIA = "criteria"


class ConversionMode(Enum):
    """Conversion modes for group operations."""
    SHORTEN = "shorten"
    EXPAND = "expand"


class HotelMappings:
    """Centralized mappings for hotel services and criteria."""
    
    # Service mappings
    SERVICE_ID_TO_GROUP = {
        1: "Tiện ích phòng nghỉ", 3: "Tiện ích phòng nghỉ", 14: "Tiện ích phòng nghỉ",
        2: "Tiện ích công nghệ & làm việc", 18: "Tiện ích công nghệ & làm việc",
        4: "Dịch vụ ẩm thực", 13: "Tiện ích thiên nhiên",
        5: "Tiện ích giải trí & spa", 11: "Tiện ích giải trí & spa", 16: "Tiện ích giải trí & spa",
        6: "Tiện ích dành cho trẻ em",
        7: "Dịch vụ & cơ sở vật chất chung", 15: "Dịch vụ & cơ sở vật chất chung",
        8: "Dịch vụ bảo mật & an toàn",
        9: "Dịch vụ sự kiện & hỗ trợ đặc biệt", 10: "Dịch vụ sự kiện & hỗ trợ đặc biệt",
        12: "Dịch vụ vận chuyển & bãi đỗ xe", 17: "Dịch vụ vận chuyển & bãi đỗ xe"
    }
    
    SERVICE_TEXT_TO_GROUP = {
        "Tiện ích phòng": "Tiện ích phòng nghỉ",
        "Tiện ích công nghệ & kết nối": "Tiện ích công nghệ & làm việc",
        "Tiện ích phòng tắm": "Tiện ích phòng nghỉ",
        "Dịch vụ ẩm thực": "Dịch vụ ẩm thực",
        "Tiện ích thư giãn & giải trí": "Tiện ích giải trí & spa",
        "Tiện ích dành cho trẻ em": "Tiện ích dành cho trẻ em",
        "Dịch vụ khách sạn": "Dịch vụ & cơ sở vật chất chung",
        "Dịch vụ bảo mật & an toàn": "Dịch vụ bảo mật & an toàn",
        "Dịch vụ tổ chức sự kiện": "Dịch vụ sự kiện & hỗ trợ đặc biệt",
        "Tiện nghi hỗ trợ đặc biệt": "Dịch vụ sự kiện & hỗ trợ đặc biệt",
        "Dịch vụ giải trí ngoài trời": "Tiện ích giải trí & spa",
        "Dịch vụ bãi đỗ xe": "Dịch vụ vận chuyển & bãi đỗ xe",
        "Tiện ích thiên nhiên": "Tiện ích giải trí & spa",
        "Hệ thống điều hòa & sưởi ấm": "Tiện ích phòng nghỉ",
        "Cơ sở vật chất chung": "Dịch vụ & cơ sở vật chất chung",
        "Dịch vụ chăm sóc sức khỏe & spa": "Tiện ích giải trí & spa",
        "Dịch vụ đưa đón & phương tiện di chuyển": "Dịch vụ vận chuyển & bãi đỗ xe",
        "Tiện ích công việc": "Tiện ích công nghệ & làm việc"
    }
    
    SERVICE_GROUP_TO_TEXTS = {
        "Tiện ích phòng nghỉ": ["Tiện ích phòng", "Tiện ích phòng tắm", "Hệ thống điều hòa & sưởi ấm"],
        "Tiện ích công nghệ & làm việc": ["Tiện ích công nghệ & kết nối", "Tiện ích công việc"],
        "Dịch vụ ẩm thực": ["Dịch vụ ẩm thực"],
        "Tiện ích giải trí & spa": ["Tiện ích thư giãn & giải trí", "Dịch vụ giải trí ngoài trời", 
                                    "Dịch vụ chăm sóc sức khỏe & spa"],
        "Tiện ích dành cho trẻ em": ["Tiện ích dành cho trẻ em"],
        "Dịch vụ & cơ sở vật chất chung": ["Dịch vụ khách sạn", "Cơ sở vật chất chung"],
        "Dịch vụ bảo mật & an toàn": ["Dịch vụ bảo mật & an toàn"],
        "Dịch vụ sự kiện & hỗ trợ đặc biệt": ["Dịch vụ tổ chức sự kiện", "Tiện nghi hỗ trợ đặc biệt"],
        "Dịch vụ vận chuyển & bãi đỗ xe": ["Dịch vụ bãi đỗ xe", "Dịch vụ đưa đón & phương tiện di chuyển"]
    }
    
    # Criteria mappings
    CRITERIA_ID_TO_GROUP = {
        1: "Vị trí & môi trường", 14: "Vị trí & môi trường", 15: "Vị trí & môi trường",
        2: "Phòng nghỉ & tiện nghi", 4: "Phòng nghỉ & tiện nghi", 
        9: "Phòng nghỉ & tiện nghi", 13: "Phòng nghỉ & tiện nghi",
        3: "Sạch sẽ",
        5: "Dịch vụ & nhân viên", 6: "Dịch vụ & nhân viên",
        7: "Wi-Fi",
        8: "Ẩm thực",
        10: "Cơ sở vật chất", 16: "Cơ sở vật chất",
        11: "Giá trị",
        12: "An ninh",
    }
    
    CRITERIA_TEXT_TO_GROUP = {
        "location": "Vị trí & môi trường",
        "room": "Phòng nghỉ & tiện nghi",
        "cleanliness": "Sạch sẽ",
        "comfort": "Phòng nghỉ & tiện nghi",
        "service": "Dịch vụ & nhân viên",
        "staff": "Dịch vụ & nhân viên",
        "wifi": "Wi-Fi",
        "food": "Ẩm thực",
        "bathroom": "Phòng nghỉ & tiện nghi",
        "parking": "Cơ sở vật chất",
        "value": "Giá trị",
        "facilities": "Cơ sở vật chất",
        "air conditioning": "Phòng nghỉ & tiện nghi",
        "view": "Vị trí & môi trường",
        "environment": "Vị trí & môi trường",
        "security": "An ninh"
    }
    
    CRITERIA_GROUP_TO_TEXTS = {
        "Vị trí & môi trường": ["location", "view", "environment"],
        "Phòng nghỉ & tiện nghi": ["room", "comfort", "bathroom", "air conditioning"],
        "Sạch sẽ": ["cleanliness"],
        "Dịch vụ & nhân viên": ["service", "staff"],
        "Wi-Fi": ["wifi"],
        "Ẩm thực": ["food"],
        "Cơ sở vật chất": ["facilities", "parking"],
        "An ninh": ["security"],
        "Giá trị": ["value"]
    }
    
    # Amenities mapping
    AMENITIES = {
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


class HotelUtilities:
    """Utility class for hotel recommendation system operations."""
    
    def __init__(self):
        self.mappings = HotelMappings()
    
    def convert_group(self, 
                     input_value: Union[int, str], 
                     category: Category = Category.SERVICES, 
                     mode: ConversionMode = ConversionMode.SHORTEN) -> Union[str, List[str], None]:
        """
        Convert between service or criteria ID and text, or expand a group to a list of texts.

        Args:
            input_value: The value to convert (ID or text)
            category: The category of the input value
            mode: The conversion mode

        Returns:
            Converted value or list of values, None if not found
        """
        if mode == ConversionMode.SHORTEN:
            return self._shorten_conversion(input_value, category)
        elif mode == ConversionMode.EXPAND:
            return self._expand_conversion(input_value, category)
        return None
    
    def _shorten_conversion(self, input_value: Union[int, str], category: Category) -> Optional[str]:
        """Convert ID/text to group name."""
        if category == Category.SERVICES:
            if isinstance(input_value, int):
                return self.mappings.SERVICE_ID_TO_GROUP.get(input_value)
            elif isinstance(input_value, str):
                return self.mappings.SERVICE_TEXT_TO_GROUP.get(input_value)
        elif category == Category.CRITERIA:
            if isinstance(input_value, int):
                return self.mappings.CRITERIA_ID_TO_GROUP.get(input_value)
            elif isinstance(input_value, str):
                return self.mappings.CRITERIA_TEXT_TO_GROUP.get(input_value)
        return None
    
    def _expand_conversion(self, input_value: str, category: Category) -> List[str]:
        """Expand group name to list of texts."""
        if category == Category.SERVICES:
            return self.mappings.SERVICE_GROUP_TO_TEXTS.get(input_value, [])
        elif category == Category.CRITERIA:
            return self.mappings.CRITERIA_GROUP_TO_TEXTS.get(input_value, [])
        return []
    
    def filter_matching_elements(self, ids_a: List[int], items_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter elements in B that have IDs present in A.
        
        Args:
            ids_a: List of IDs to match
            items_b: List of items to filter
            
        Returns:
            Filtered list of items
        """
        if not items_b:
            return []
        
        # Try 'id' field first, fallback to 'id_room'
        id_field = 'id' if 'id' in items_b[0] else 'id_room'
        
        return [item for item in items_b if item.get(id_field) in ids_a]
    
    def convert_amenity(self, value: Union[int, str]) -> Union[str, int]:
        """
        Convert between amenity ID and name.

        Args:
            value: The amenity ID or name

        Returns:
            The corresponding amenity name (if input is int) or ID (if input is str)
        """
        if isinstance(value, int):
            return self.mappings.AMENITIES.get(value, "Không tìm thấy")
        elif isinstance(value, str):
            for amenity_id, amenity_name in self.mappings.AMENITIES.items():
                if amenity_name == value:
                    return amenity_id
            return "Không tìm thấy"
        return "Định dạng không hợp lệ"


class ScoreCalculator:
    """Calculator for combining and computing scores from multiple sources."""
    
    @staticmethod
    def _normalize_to_range(values_dict: Dict[Any, float], 
                           min_target: float = 0.25, 
                           max_target: float = 1.0) -> Dict[Any, float]:
        """
        Normalize values to a specified range.
        
        Args:
            values_dict: Dictionary of values to normalize
            min_target: Minimum target value
            max_target: Maximum target value
            
        Returns:
            Dictionary with normalized values
        """
        values = list(values_dict.values())
        if not values:
            return {k: min_target for k in values_dict}
        
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return {k: min_target for k in values_dict}
        
        return {
            k: min_target + (max_target - min_target) * (v - min_v) / (max_v - min_v)
            for k, v in values_dict.items()
        }
    
    @staticmethod
    def _validate_inputs(sources: Tuple[Dict], weights: Optional[List[float]]) -> List[float]:
        """Validate inputs and return proper weights."""
        if not sources:
            raise ValueError("At least one source must be provided")
        
        if not all(isinstance(source, dict) for source in sources):
            raise ValueError("All sources must be dictionaries")
        
        if weights is None:
            weights = [1.0 / len(sources)] * len(sources)
        
        if len(weights) != len(sources):
            raise ValueError("Number of weights must match number of sources")
        
        return weights
    
    def compute_total_score(self, *sources: Dict[Any, float], 
                           weights: Optional[List[float]] = None) -> List[Tuple[Any, float]]:
        """
        Compute total score for each ID by combining scores from multiple sources.

        Args:
            *sources: Multiple dictionaries containing scores for each ID
            weights: List of weights for each source

        Returns:
            List of tuples (id, score) sorted by score in descending order
        """
        weights = self._validate_inputs(sources, weights)
        normalized_sources = [self._normalize_to_range(source) for source in sources]
        
        total_scores = defaultdict(float)
        all_ids = set()
        for source in normalized_sources:
            all_ids.update(source.keys())
        
        for id_ in all_ids:
            if len(sources) == 2:
                # Special handling for two sources
                sources_with_id = sum(1 for source in sources if id_ in source)
                if sources_with_id == 1:
                    # Single source penalty
                    for source in sources:
                        if id_ in source:
                            total_scores[id_] = source[id_] * 0.8
                            break
                else:
                    # Both sources present
                    total_scores[id_] = sum(
                        source.get(id_, 0.0) * weight
                        for source, weight in zip(sources, weights)
                    )
            else:
                # Standard weighted sum for other cases
                total_scores[id_] = sum(
                    source.get(id_, 0.0) * weight
                    for source, weight in zip(sources, weights)
                )
        
        return sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
    
    def compute_intersection_score(self, *sources: Dict[Any, float], 
                                 weights: Optional[List[float]] = None) -> List[Tuple[Any, float]]:
        """
        Compute total score only for IDs present in all sources.

        Args:
            *sources: Multiple dictionaries containing scores for each ID
            weights: List of weights for each source

        Returns:
            List of tuples (id, score) sorted by score in descending order
        """
        weights = self._validate_inputs(sources, weights)
        normalized_sources = [self._normalize_to_range(source) for source in sources]
        
        # Get intersection of all IDs
        common_ids = set(normalized_sources[0].keys())
        for source in normalized_sources[1:]:
            common_ids &= set(source.keys())
        
        total_scores = {}
        for id_ in common_ids:
            total_scores[id_] = sum(
                source.get(id_, 0.0) * weight
                for source, weight in zip(normalized_sources, weights)
            )
        
        return sorted(total_scores.items(), key=lambda x: x[1], reverse=True)


# Legacy function wrappers for backward compatibility
def convert_group(input_value: Union[int, str], 
                 category: str = "services", 
                 model: str = "shorten") -> Union[str, List[str], None]:
    """Legacy wrapper for backward compatibility."""
    utils = HotelUtilities()
    cat = Category.SERVICES if category == "services" else Category.CRITERIA
    mode = ConversionMode.SHORTEN if model == "shorten" else ConversionMode.EXPAND
    return utils.convert_group(input_value, cat, mode)


def filter_matching_elements(ids_A: List[int], B: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Legacy wrapper for backward compatibility."""
    utils = HotelUtilities()
    return utils.filter_matching_elements(ids_A, B)


def convert_amenity(value: Union[int, str]) -> Union[str, int]:
    """Legacy wrapper for backward compatibility."""
    utils = HotelUtilities()
    return utils.convert_amenity(value)


def compute_total_score(*sources: Dict, weights: Optional[List[float]] = None) -> List[Tuple[Any, float]]:
    """Legacy wrapper for backward compatibility."""
    calculator = ScoreCalculator()
    return calculator.compute_total_score(*sources, weights=weights)


def compute_intersection_score(*sources: Dict, weights: Optional[List[float]] = None) -> List[Tuple[Any, float]]:
    """Legacy wrapper for backward compatibility."""
    calculator = ScoreCalculator()
    return calculator.compute_intersection_score(*sources, weights=weights)