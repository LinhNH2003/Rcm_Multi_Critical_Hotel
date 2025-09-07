# Hotel Recommendation System

Hệ thống khuyến nghị khách sạn thông minh sử dụng machine learning và semantic similarity để đưa ra gợi ý phù hợp với nhu cầu của người dùng.

## Tính năng chính

- **Khuyến nghị dựa trên tiện nghi**: Sử dụng semantic similarity để tìm khách sạn phù hợp với tiện nghi mong muốn
- **Đánh giá vị trí**: Tính toán điểm số dựa trên vị trí và khoảng cách đến các địa điểm quan tâm
- **Phân tích giá cả**: Đánh giá mức độ phù hợp về giá cả
- **Đánh giá chất lượng**: Sử dụng Bayesian averaging để đánh giá chất lượng khách sạn
- **Phân tích chính sách**: So sánh và đánh giá tính tương thích của các chính sách khách sạn
- **Web scraping**: Thu thập dữ liệu từ các trang web đặt phòng khách sạn

## Cấu trúc dự án

```
hotel-recommendation-system/
├── config/                 # Cấu hình hệ thống
│   ├── __init__.py
│   ├── settings.py         # Cài đặt chính
│   └── constants.py        # Hằng số và enum
├── schemas/                # Data models và schemas
│   ├── __init__.py
│   └── models.py           # Định nghĩa các model dữ liệu
├── services/               # Business logic
│   ├── __init__.py
│   └── recommendation_service.py  # Dịch vụ khuyến nghị
├── modules/                # Các module chức năng
│   ├── recommend_amenity.py
│   ├── recommend_location.py
│   ├── recommend_policies.py
│   ├── recommend_pricing.py
│   ├── recommend_rating.py
│   ├── recommend_review.py
│   ├── recommned_room.py
│   └── review_quality.py
├── object/                 # Object definitions
│   └── query.py
├── scripts/                # Scripts tiện ích
│   └── scraper_booking.py
├── testing/                # Testing và analysis
│   └── bayesian_average.py
├── utils/                  # Utilities
│   └── utils.py
├── data/                   # Dữ liệu (tự tạo)
├── models/                 # Models đã train (tự tạo)
├── logs/                   # Log files (tự tạo)
├── cache/                  # Cache files (tự tạo)
├── output/                 # Kết quả output (tự tạo)
├── main.py                 # Entry point chính
├── requirements.txt        # Dependencies
└── README.md              # Hướng dẫn này
```

## Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd hotel-recommendation-system
```

### 2. Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate      # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình môi trường

Tạo file `.env` trong thư mục gốc:

```env
# Database settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME=hotel_recommendation
DB_USER=postgres
DB_PASSWORD=your_password

# Model settings
MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
USE_GPU=true
MODEL_DIR=./models

# Scraping settings
HEADLESS=true
SCRAPING_TIMEOUT=30
```

## Sử dụng

### Chế độ Interactive

Chạy ứng dụng ở chế độ tương tác:

```bash
python main.py --mode interactive
```

Hệ thống sẽ hướng dẫn bạn nhập:
- Thông tin địa điểm
- Khoảng giá
- Tiện nghi mong muốn
- Thông tin phòng
- Tùy chọn đặt phòng

### Chế độ Batch

Chạy với file query có sẵn:

```bash
python main.py --mode batch --query query.json --data hotel_data.json --output results.json
```

#### Ví dụ file query.json:

```json
{
  "province": "Lâm Đồng",
  "nearby_places": ["Chợ Đà Lạt", "Hồ Xuân Hương"],
  "is_near_center": true,
  "price_range": [300000, 800000],
  "stars_rating": 3,
  "amenities": ["TV", "máy sưởi"],
  "capacity": 2,
  "room_type": "Double Room",
  "included_breakfast": false
}
```

### Sử dụng API

```python
from schemas import HotelQuery
from services import RecommendationService, DataService

# Tạo query
query_data = {
    "province": "Hà Nội",
    "amenities": ["WiFi", "Parking"],
    "price_range": [500000, 1500000],
    "capacity": 2
}
query = HotelQuery.from_dict(query_data)

# Load dữ liệu và tạo recommendations
data_service = DataService()
hotel_data = data_service.load_hotel_data("hotel_data.json")

recommendation_service = RecommendationService()
recommendations = recommendation_service.recommend_hotels(query, hotel_data)

# Hiển thị kết quả
for rec in recommendations[:5]:
    print(f"Hotel: {rec.hotel_id}, Score: {rec.final_score:.3f}")
```

## Các module chính

### 1. Amenity Recommendation (`modules/recommend_amenity.py`)

Sử dụng semantic similarity để tìm khách sạn phù hợp với tiện nghi:

```python
from modules.recommend_amenity import HotelSimilarityRecommender

recommender = HotelSimilarityRecommender()
recommender.fit(hotel_data)
results = recommender.predict_top_k(["WiFi", "Parking"], top_k=10)
```

### 2. Location Scoring (`modules/recommend_location.py`)

Tính điểm dựa trên vị trí và khoảng cách:

```python
from modules.recommend_location import LocationScorer

scorer = LocationScorer()
score = scorer.calculate_location_score(
    hotel, 
    user_places=["Chợ Bến Thành", "Phố đi bộ Nguyễn Huệ"],
    province="TP Hồ Chí Minh"
)
```

### 3. Policy Analysis (`modules/recommend_policies.py`)

Phân tích và so sánh chính sách khách sạn:

```python
from modules.recommend_policies import PolicyScorer

scorer = PolicyScorer()
score = scorer.find_similar_hotel_policies(user_policies, hotel_policies)
```

### 4. Price Analysis (`modules/recommend_pricing.py`)

Đánh giá mức độ phù hợp về giá:

```python
from modules.recommend_pricing import PriceScorer

scorer = PriceScorer(user_input={"price_range": (500000, 1000000)})
score = scorer.score_room(room_data)
```

### 5. Rating Analysis (`modules/recommend_rating.py`)

Phân tích đánh giá và điểm số:

```python
from modules.recommend_rating import RatingScorer

scorer = RatingScorer(user_input={"stars_rating": 4, "rating": 8.5})
results = scorer.score_hotels(hotels)
```

### 6. Review Quality (`modules/review_quality.py`)

Phân tích chất lượng đánh giá:

```python
from modules.review_quality import HotelReviewQualityAnalyzer

analyzer = HotelReviewQualityAnalyzer(reviews)
analyzer.process_reviews()
scores = analyzer.calculate_global_scores()
```

## Web Scraping

### Thu thập dữ liệu từ Booking.com

```python
from scripts.scraper_booking import scrape_hotel_data

# Scrape thông tin khách sạn
hotel_data, info_success, review_success = scrape_hotel_data(
    url="https://www.booking.com/hotel/vn/example.html",
    IS_GET_INFO_HOTEL=True,
    IS_GET_INFOMATION_ROOM=True,
    IS_GET_POLICY=True,
    IS_GET_REVIEW=True
)
```

## Testing và Analysis

### Bayesian Analysis

```python
from testing.bayesian_average import run_bayesian_analysis

analyzer = run_bayesian_analysis(
    data_path="review_quality.json",
    save_plots=True,
    plot_dir="./plots"
)
```

## Cấu hình nâng cao

### Custom Configuration

Tạo file `config/custom_settings.py`:

```python
from config.settings import settings

# Override settings
settings.scoring.similarity_threshold = 0.85
settings.model.batch_size = 256
settings.scraping.timeout = 60
```

### Database Integration

```python
from config.settings import settings

# Sử dụng database connection
connection_string = settings.database.connection_string
```

## Troubleshooting

### Lỗi thường gặp

1. **ModuleNotFoundError**: Đảm bảo đã cài đặt đầy đủ dependencies
2. **CUDA out of memory**: Giảm batch_size hoặc sử dụng CPU
3. **File not found**: Kiểm tra đường dẫn file dữ liệu
4. **Chrome driver issues**: Cập nhật ChromeDriverManager

### Debug mode

```bash
python main.py --mode interactive --config debug_config.json
```

## Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request


