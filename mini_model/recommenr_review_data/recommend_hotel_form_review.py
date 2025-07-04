from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
import sys
import os
import json


def allocate_weights_with_ratios(selected_weights: dict = None, total_selected_weight: float = 0.0):

    """
    Phân bổ trọng số cho các tiêu chí đánh giá

    selected_weights: dict dạng {'location': 2, 'room': 1} => location gấp đôi room
    total_selected_weight: tổng trọng số phân bổ cho các key được chọn 
    Returns:
        Dictionary chứa trọng số phân bổ cho các key
    """
    all_keys = [
        'location', 'room', 'cleanliness', 'comfort', 'service',
        'staff', 'wifi', 'food', 'bathroom', 'parking',
        'value', 'facilities', 'air conditioning', 'view',
        'environment', 'security'
    ]
    

    if selected_weights is None:
        # Trường hợp không có key nào được chọn → chia đều 1/16
        equal_weight = 1.0 / len(all_keys)
        return {key: equal_weight for key in all_keys}

    if total_selected_weight == 0:
        total_selected_weight = sum(selected_weights.values()) / 10
        
    total_ratio = sum(selected_weights.values())
    per_unit_weight = total_selected_weight / total_ratio

    result = {}
    for key in all_keys:
        if key in selected_weights:
            # Gán theo tỉ lệ
            result[key] = selected_weights[key] * per_unit_weight
        else:
            result[key] = 0  # Tạm thời gán 0 để sau cộng thêm phần chia đều
    
    # Tính phần còn lại và chia đều cho 16 key
    total_allocated = sum(result.values())
    remainder_per_key = (1 - total_allocated) / len(all_keys)
    
    # Cộng phần còn lại vào tất cả key
    for key in result:
        result[key] += remainder_per_key

    return result


def calculate_weighted_bayesian_score(result_groupby_id, user_weights: dict = None, default_weight=1/16, C=100, print_process=False):
    """
    Tính điểm có trọng số và điều chỉnh Bayesian cho các khách sạn dựa trên trọng số người dùng.

    Parameters:
    - result_groupby_id: Dictionary chứa dữ liệu khách sạn {hotel_id: {'summary': {...}}}
    - user_weights: Dictionary chứa trọng số do người dùng chỉ định {criterion: weight}
    - default_weight: Trọng số mặc định cho tiêu chí không được chỉ định (mặc định 0.1)
    - C: Hằng số siêu tham số cho Bayesian (mặc định 100)

    Returns:
    - DataFrame với các cột ['hotel_id', 'review_count', 'weighted_score', 
                            'normalized_weighted_score', 'bayesian_score', 
                            'positive_rates', 'weights']
    """
    # Step 1: Tính điểm có trọng số cho mỗi khách sạn
    results = []
    
    for hotel_id, data in result_groupby_id.items():
        criteria_summary = data['summary']['criteria_summary']
        review_count = data['summary']['review_count']
        avg_rating = data['summary']['avg_rating']
        
        # Lấy positive rates và total reviews
        positive_rates = {}
        total_reviews = {}
        for criterion, ratings in criteria_summary.items():
            positive_rates[criterion] = ratings['Positive']
            total_reviews[criterion] = ratings['total']
        
        if user_weights is None:
            weights = {criterion: count / (review_count * 10) for criterion, count in total_reviews.items()}
        else:
            weights = {}
            # total_weight = 0
            for criterion in criteria_summary:
                weights[criterion] = user_weights.get(criterion, default_weight)
                
        # Chuẩn hóa điểm nếu tổng trọng số != 1
        # Step 3: Compute raw score
        raw_score = 0
        for criterion in criteria_summary:
            raw_score += weights[criterion] * positive_rates[criterion]
        
        
        # Lưu kết quả cho khách sạn
        results.append({
            'hotel_id': hotel_id,
            'review_count': review_count,
            'avg_rating': avg_rating,
            'normalized_weighted_score': raw_score,
            'positive_rates': positive_rates,
            'total_reviews': total_reviews,
            'weights': weights
        })
    
    # Chuyển kết quả thành DataFrame
    df_results = pd.DataFrame(results)
    
    # Step 2: Tính mu (global average positive rate)
    total_positive_reviews = 0
    total_reviews = 0
    
    for hotel_id,  data in result_groupby_id.items():
        criteria_summary = data['summary']['criteria_summary']
        for criterion, ratings in criteria_summary.items():
            positive_reviews = ratings['Positive'] * ratings['total']
            total_positive_reviews += positive_reviews
            total_reviews += ratings['total']
    
    mu = total_positive_reviews / total_reviews if total_reviews > 0 else 0
    if print_process:
        print(f"Total positive reviews: {total_positive_reviews}, Total reviews: {total_reviews}, mu: {mu}")
    
    # Step 3: Áp dụng điều chỉnh Bayesian
    df_results['bayesian_score'] = df_results.apply(
        lambda row: (row['review_count'] * row['normalized_weighted_score'] + C * mu) / 
                    (row['review_count'] + C) if row['review_count'] + C > 0 else mu,
        axis=1
    )
    
    return df_results

def calculate_final_score_from_reviews(score_review, score_review_quality, w=0.8, q_default=0.5, s_global=0.5, threshold=0.00):
    """
    Tính điểm cuối cùng (F_i) cho khách sạn từ hai dictionary score_review và score_review_quality.
    
    Parameters:
    - score_review: Dictionary chứa recommend_score (S_i^adj) {hotel_id: score}
    - score_review_quality: Dictionary chứa quality_score (Q_i) {hotel_id: score}
    - w: Trọng số cho Q_i (mặc định 0.8)
    - q_default: Điểm chất lượng mặc định (mặc định 0.5)
    - s_global: Điểm recommend_score trung bình toàn cục (mặc định 0.55)
    - threshold: Ngưỡng cho Q_i (mặc định 0.00)
    
    Returns:
    - DataFrame với các cột ['id', 'recommend_score', 'quality_score', 'final_score'], 
      sắp xếp theo final_score giảm dần
    """
    # Tạo danh sách các hotel_id có trong ít nhất một dictionary
    hotel_ids = list(set(score_review.keys()) | set(score_review_quality.keys()))
    
    # Tạo DataFrame từ các dictionary
    data = {
        'id': hotel_ids,
        'recommend_score': [score_review.get(hid, np.nan) for hid in hotel_ids],
        'quality_score': [score_review_quality.get(hid, np.nan) for hid in hotel_ids]
    }
    df = pd.DataFrame(data)
    
    # Khởi tạo cột final_score
    df['final_score'] = np.nan
    
    # Hàm tính F_i cho mỗi hàng
    def compute_final_score(row):
        s_adj = row['recommend_score']
        q_i = row['quality_score']
        
        # Trường hợp 1: Không có đánh giá (s_adj là NaN)
        if pd.isna(s_adj):
            return s_global * q_default
        
        # Trường hợp 2: Không có quality_score (q_i là NaN)
        if pd.isna(q_i):
            return s_adj * (1 - w) * q_default
        
        # Trường hợp 3: Q_i < threshold
        if q_i < threshold:
            return s_global * q_default
        
        # Trường hợp 4: Q_i = 0
        if q_i == 0:
            return s_adj * (1 - w) * q_default
        
        # Trường hợp thông thường: Damped Multiplicative Combination
        return s_adj * (w * q_i + (1 - w) * q_default)
    
    # Áp dụng hàm tính F_i
    df['final_score'] = df.apply(compute_final_score, axis=1)
    
    # Sắp xếp theo final_score giảm dần
    df = df.sort_values(by='final_score', ascending=False)
    
    return df

