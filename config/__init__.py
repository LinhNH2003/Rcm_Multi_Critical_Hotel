"""
Configuration package for hotel recommendation system.
"""

from .settings import settings, get_path
from .constants import *

__all__ = [
    'settings',
    'get_path',
    'RoomType',
    'RoomLevel', 
    'BedType',
    'ServiceCategory',
    'QualityCriteria',
    'ScoringMethod',
    'ConversionMode',
    'Category',
    'SERVICE_ID_TO_GROUP',
    'SERVICE_TEXT_TO_GROUP',
    'SERVICE_GROUP_TO_TEXTS',
    'CRITERIA_ID_TO_GROUP',
    'CRITERIA_TEXT_TO_GROUP',
    'CRITERIA_GROUP_TO_TEXTS',
    'AMENITIES',
    'ROOM_TYPE_LIST',
    'ROOM_LEVEL_LIST',
    'BED_TYPE_LIST',
    'ALL_CRITERIA',
    'STANDARD_PROVINCES',
    'DEFAULT_WEIGHTS',
    'DYNAMIC_WEIGHTS_HIGH',
    'LONG_REVIEW_THRESHOLD',
    'IMAGE_RATIO_THRESHOLD',
    'LONG_REVIEW_RATIO_THRESHOLD',
    'POLICY_PATTERNS',
    'SUB_RATING_KEY_MAP',
    'DEFAULT_CONFIG'
]

