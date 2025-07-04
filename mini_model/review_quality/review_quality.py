import json
import math
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

import sys
from pathlib import Path

# Thêm thư mục gốc (mini_model) vào sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config

class HotelReviewStats:
    def __init__(self, reviews=None):
        self.reviews = reviews
        self.results = []
        self.stats = defaultdict(lambda: {
            "count_review": 0,
            "count_review_with_image": 0,
            "count_long_review": 0,
            "total_score": 0.0,
            "num_score": 0,
            "avg_score": 0.0
        })
        self.final_scores = {}

    def process_reviews(self):
        """
        Xử lý danh sách đánh giá và tính toán các số liệu thống kê cơ bản.
        """
        for review in tqdm(self.reviews, total=len(self.reviews), desc="Processing: "):
            hotel_id = review.get('id')
            if not hotel_id:  # Kiểm tra nếu hotel_id không tồn tại
                continue
            self.stats[hotel_id]['count_review'] += 1
            
            # Review with image
            if review.get('review_photo'):
                self.stats[hotel_id]['count_review_with_image'] += 1
                
            # Long review
            pos = review.get("review_positive", "")
            neg = review.get("review_negative", "")
            text = f"{pos} {neg}".strip()

            if len(text) > 150:
                self.stats[hotel_id]['count_long_review'] += 1
            
            score_str = review.get("review_score")
            try:
                score = float(score_str.split()[-1].replace(",", "."))
                self.stats[hotel_id]["total_score"] += score
                self.stats[hotel_id]["num_score"] += 1
                self.stats[hotel_id]["avg_score"] = (
                    self.stats[hotel_id]["total_score"] / self.stats[hotel_id]["num_score"]
                )
            except:
                continue
        
        return self.stats

    def save_stats(self, filename='review_quality.json.json'):
        """
        Lưu self.stats vào file JSON.
        
        Args:
            filename: Tên file để lưu (mặc định: hotel_stats.json)
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dict(self.stats), f, ensure_ascii=False, indent=4)
        print(f"Stats saved to {filename}")

    def load_stats(self, filename=config.get_path("review_quality.json")):
        """
        Load self.stats từ file JSON.
        
        Args:
            filename: Tên file để đọc (mặc định: review_quality.json.json)
        
        Raises:
            FileNotFoundError: Nếu file không tồn tại
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")
        
        with open(filename, 'r', encoding='utf-8') as f:
            loaded_stats = json.load(f)
        
        # Chuyển dictionary thường thành defaultdict
        self.stats = defaultdict(lambda: {
            "count_review": 0,
            "count_review_with_image": 0,
            "count_long_review": 0,
            "total_score": 0.0,
            "num_score": 0,
            "avg_score": 0.0
        })

        for hotel_id, data in loaded_stats.items():
            self.stats[hotel_id] = data
        
        print(f"Stats loaded from {filename}")
        return self.stats
        

    def calculate_statistics(self, hotel_ids=None):
        """
        Tính các số liệu thống kê cho các tham số trong stats.
        Nếu hotel_ids được cung cấp, chỉ tính trên các khách sạn được chỉ định.
        
        Args:
            hotel_ids: List chứa các hotel_id (mặc định: None, tính trên tất cả)
        
        Returns:
            Dictionary chứa số liệu thống kê
        """
        # Nếu hotel_ids được cung cấp, lọc stats
        if hotel_ids is not None:
            filtered_stats = {hid: self.stats[hid] for hid in hotel_ids if hid in self.stats}
        else:
            filtered_stats = self.stats
        
        count_reviews = []
        count_reviews_with_image = []
        count_long_reviews = []
        avg_scores = []
        
        for hotel_id, data in filtered_stats.items():
            count_reviews.append(data['count_review'])
            count_reviews_with_image.append(data['count_review_with_image'])
            count_long_reviews.append(data['count_long_review'])
            avg_scores.append(data['avg_score'])
        
        def compute_stats(values):
            return {
                'count': len(values),
                'mean': np.mean(values) if values else 0,
                'std': np.std(values) if len(values) > 1 else 0,
                'min': np.min(values) if values else 0,
                'max': np.max(values) if values else 0
            }
        
        statistics = {
            'count_review': compute_stats(count_reviews),
            'count_review_with_image': compute_stats(count_reviews_with_image),
            'count_long_review': compute_stats(count_long_reviews),
            'avg_score': compute_stats(avg_scores)
        }
        
        return statistics
        
    def calculate_final_score_local_data(self, hotel_ids, print_warning=False):
        """
        Tính điểm số (0-10) cho danh sách khách sạn được chỉ định, sử dụng số liệu thống kê tính lại
        với trọng số động dựa trên count_review.

        Args:
            hotel_ids: List chứa các hotel_id cần tính điểm
            print_warning: In cảnh báo nếu hotel_id không hợp lệ

        Returns:
            Dictionary với hotel_id và final_score
        """
        # Kiểm tra nếu danh sách rỗng hoặc không có hotel_id hợp lệ
        valid_ids = [hid for hid in hotel_ids if hid in self.stats]
        if not valid_ids:
            if print_warning:
                print("Warning: No valid hotel IDs provided")
            return {}

        # Tính số liệu thống kê trên tập con hotel_ids
        stats_info = self.calculate_statistics(hotel_ids=valid_ids)

        # Lấy ngưỡng count_review (trung bình)
        count_review_threshold = stats_info['count_review']['mean']

        local_scores = {}

        for hotel_id in hotel_ids:
            if hotel_id not in self.stats:
                if print_warning:
                    print(f"Warning: Hotel ID {hotel_id} not found in stats")
                continue

            data = self.stats[hotel_id]
            # Lấy các tham số
            count_review = data['count_review']
            count_review_with_image = data['count_review_with_image']
            count_long_review = data['count_long_review']
            avg_score = data['avg_score']

            # Chuẩn hóa các tham số
            # 1. avg_score: Chuẩn hóa tuyến tính về 0-1 dựa trên min/max
            norm_avg_score = (avg_score - stats_info['avg_score']['min']) / (
                stats_info['avg_score']['max'] - stats_info['avg_score']['min']
            ) if stats_info['avg_score']['max'] != stats_info['avg_score']['min'] else 0

            # 2. count_review: Chuẩn hóa logarit để giảm ảnh hưởng giá trị lớn
            max_log_review = math.log1p(stats_info['count_review']['max'])
            norm_count_review = math.log1p(count_review) / max_log_review if count_review > 0 else 0

            # 3. image_ratio: Chuẩn hóa với ngưỡng 0.5
            image_ratio = count_review_with_image / count_review if count_review > 0 else 0
            norm_review_with_image = min(image_ratio / 0.5, 1.0)

            # 4. long_review_ratio: Chuẩn hóa với ngưỡng 0.5
            long_review_ratio = count_long_review / count_review if count_review > 0 else 0
            norm_long_review = min(long_review_ratio / 0.5, 1.0)

            # Trọng số động dựa trên count_review
            if count_review > count_review_threshold:
                # Khi count_review lớn, tăng trọng số avg_score, giảm count_review
                w_avg_score = 0.3
                w_count_review = 0.5
            else:
                # Khi count_review nhỏ, giữ trọng số ban đầu
                w_avg_score = 0.2
                w_count_review = 0.6

            # Trọng số cho image và long review giữ nguyên
            w_image = 0.1
            w_long_review = 0.1

            # Tính final_score
            final_score = (
                w_avg_score * norm_avg_score +
                w_count_review * norm_count_review +
                w_image * norm_review_with_image +
                w_long_review * norm_long_review
            )


            local_scores[hotel_id] = round(final_score, 2)

        return local_scores

    def calculate_final_score_global_data(self):
        
        """
        Tính điểm số cuối cùng (0-10) cho mỗi khách sạn dựa trên số liệu thống kê.
        
        Returns:
            Dictionary với hotel_id và final_score
        """
        # Lấy số liệu thống kê để chuẩn hóa
        stats_info = self.calculate_statistics()
        
        # Trích xuất mean và std để chuẩn hóa z-score
        mean_count_review = stats_info['count_review']['mean']
        std_count_review = stats_info['count_review']['std'] if stats_info['count_review']['std'] > 0 else 1
        mean_review_with_image = stats_info['count_review_with_image']['mean']
        std_review_with_image = stats_info['count_review_with_image']['std'] if stats_info['count_review_with_image']['std'] > 0 else 1
        mean_long_review = stats_info['count_long_review']['mean']
        std_long_review = stats_info['count_long_review']['std'] if stats_info['count_long_review']['std'] > 0 else 1
        mean_avg_score = stats_info['avg_score']['mean']
        std_avg_score = stats_info['avg_score']['std'] if stats_info['avg_score']['std'] > 0 else 1
        
        self.final_scores = {}
        
        for hotel_id, data in self.stats.items():
            # Lấy các tham số
            count_review = data['count_review']
            count_review_with_image = data['count_review_with_image']
            count_long_review = data['count_long_review']
            avg_score = data['avg_score']
            
            # Chuẩn hóa các tham số
            # 1. avg_score: Chuẩn hóa tuyến tính về 0-1 dựa trên min/max
            norm_avg_score = (avg_score - stats_info['avg_score']['min']) / (
                stats_info['avg_score']['max'] - stats_info['avg_score']['min']
            )
            
            # 2. count_review: Chuẩn hóa logarit để giảm ảnh hưởng giá trị lớn
            max_log_review = math.log1p(stats_info['count_review']['max'])
            norm_count_review = math.log1p(count_review) / max_log_review if count_review > 0 else 0
            
            image_ratio = (
                count_review_with_image / count_review if count_review > 0 else 0
            )
            norm_review_with_image = min(image_ratio / 0.5, 1.0)  

            long_review_ratio = (
                count_long_review / count_review if count_review > 0 else 0
            )
            norm_long_review = min(long_review_ratio / 0.5, 1.0)  
            
            final_score = (
                0.2 * norm_avg_score +          
                0.6 * norm_count_review +     
                0.1 * norm_review_with_image + 
                0.1 * norm_long_review        
            )
            
            # Chuẩn hóa về thang 0-10
            final_score = final_score * 10.0
            
            self.final_scores[hotel_id] = round(final_score, 2)
        
        return self.final_scores

    def visualize_distributions(self, output_file='distributions.png'):
        """
        Vẽ biểu đồ phân bố cho các tham số.
        
        Args:
            output_file: Tên file để lưu biểu đồ
        """
        plt.figure(figsize=(12, 8))
        
        # Lấy dữ liệu
        count_reviews = [data['count_review'] for data in self.stats.values()]
        count_reviews_with_image = [data['count_review_with_image'] for data in self.stats.values()]
        count_long_reviews = [data['count_long_review'] for data in self.stats.values()]
        avg_scores = [data['avg_score'] for data in self.stats.values()]
        
        # Vẽ 4 biểu đồ phân bố
        plt.subplot(2, 2, 1)
        plt.hist(count_reviews, bins=20, color='skyblue', edgecolor='black')
        plt.title('Distribution of Count Review')
        plt.xlabel('Count Review')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 2)
        plt.hist(count_reviews_with_image, bins=20, color='lightgreen', edgecolor='black')
        plt.title('Distribution of Count Review with Image')
        plt.xlabel('Count Review with Image')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 3)
        plt.hist(count_long_reviews, bins=20, color='salmon', edgecolor='black')
        plt.title('Distribution of Count Long Review')
        plt.xlabel('Count Long Review')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 4)
        plt.hist(avg_scores, bins=20, color='lightcoral', edgecolor='black')
        plt.title('Distribution of Avg Score')
        plt.xlabel('Avg Score')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        plt.savefig(output_file)
        plt.close()

    def visualize_scatter(self, output_file='scatter.png'):
        """
        Vẽ biểu đồ phân tán giữa avg_score và count_review, với kích thước điểm tỷ lệ với count_long_review.
        
        Args:
            output_file: Tên file để lưu biểu đồ
        """
        plt.figure(figsize=(10, 6))
        
        # Lấy dữ liệu
        count_reviews = [data['count_review'] for data in self.stats.values()]
        avg_scores = [data['avg_score'] for data in self.stats.values()]
        count_long_reviews = [data['count_long_review'] for data in self.stats.values()]
        
        # Chuẩn hóa count_long_review để làm kích thước điểm
        max_long_review = max(count_long_reviews) if count_long_reviews else 1
        sizes = [(cl / max_long_review) * 1000 for cl in count_long_reviews]
        
        plt.scatter(count_reviews, avg_scores, s=sizes, alpha=0.5, c='blue')
        plt.title('Scatter: Avg Score vs Count Review')
        plt.xlabel('Count Review')
        plt.ylabel('Avg Score')
        plt.grid(True)
        plt.show()
        plt.savefig(output_file)
        plt.close()

    def visualize_final_scores(self, output_file='final_scores.png'):
        """
        Vẽ biểu đồ phân bố của final_score.
        
        Args:
            output_file: Tên file để lưu biểu đồ
        """
        if not self.final_scores:
            self.calculate_final_score()
        
        plt.figure(figsize=(8, 6))
        scores = list(self.final_scores.values())
        plt.hist(scores, bins=20, color='purple', edgecolor='black')
        plt.title('Distribution of Final Scores')
        plt.xlabel('Final Score')
        plt.ylabel('Frequency')
        plt.show()  
        plt.savefig(output_file)
        plt.close()