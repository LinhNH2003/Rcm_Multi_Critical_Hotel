import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import parafac
import pickle
import os

class CF_user_scores:
    def __init__(self):
        self.num_criteria = 16
        self.rank = 50
        self.criteria = [
            'location_Positive', 'room_Positive', 'cleanliness_Positive', 'comfort_Positive',
            'service_Positive', 'staff_Positive', 'wifi_Positive', 'food_Positive',
            'bathroom_Positive', 'parking_Positive', 'value_Positive', 'facilities_Positive',
            'air conditioning_Positive', 'view_Positive', 'environment_Positive', 'security_Positive'
        ]
        self.statuses = ['', 'Cặp đôi', 'Khách lẻ', 'Nhóm', 'Phòng gia đình']
        #'location', 'room', 'cleanliness', 'comfort', 'service', 'staff', 'wifi', 'food', 'bathroom', 'parking', 'value', 'facilities', 'air conditioning', 'view', 'environment', 'security'

    def create_tensor_from_df(self, df, dimension='country'):
        """
        Tạo tensor từ DataFrame dựa trên quốc gia hoặc trạng thái.
        
        Args:
            df: DataFrame chứa dữ liệu
            dimension: 'country' hoặc 'state'
            
        Returns:
            tuple: tensor, dimension_to_idx, hotel_to_idx
        """
        dim_column = 'country' if dimension == 'country' else 'state'
        dimensions = sorted(df[dim_column].unique())
        hotels = sorted(df['id'].unique())
        
        num_dims_actual = len(dimensions)
        num_hotels_actual = len(hotels)
        
        dim_to_idx = {d: i for i, d in enumerate(dimensions)}
        hotel_to_idx = {h: i for i, h in enumerate(hotels)}
        
        tensor = np.full((num_dims_actual, num_hotels_actual, self.num_criteria), np.nan)
        
        for _, row in df.iterrows():
            dim_idx = dim_to_idx[row[dim_column]]
            hotel_idx = hotel_to_idx[row['id']]
            for crit_idx, crit in enumerate(self.criteria):
                tensor[dim_idx, hotel_idx, crit_idx] = row[crit]
        
        return tensor, dim_to_idx, hotel_to_idx

    def preprocess_tensor(self, tensor):
        """Điền giá trị thiếu bằng trung bình của mỗi tiêu chí."""
        for k in range(tensor.shape[2]):
            slice_k = tensor[:, :, k]
            mean_k = np.nanmean(slice_k)
            slice_k[np.isnan(slice_k)] = mean_k
            tensor[:, :, k] = slice_k
        return tensor

    def predict_scores(self, dim_idx, hotel_idx, factors, weights):
        """Dự đoán điểm số cho một cặp (dimension, hotel)."""
        u, v, w = factors
        scores = np.zeros(self.num_criteria)
        for r in range(len(weights)):
            scores += weights[r] * u[dim_idx, r] * v[hotel_idx, r] * w[:, r]
        return np.clip(scores, 0, 1)

    def save_model(self, weights, factors, dim_to_idx, hotel_to_idx, filename='model.pkl', dimension='country'):
        """Lưu model vào file."""
        model_data = {
            'weights': weights,
            'factors': factors,
            f"{dimension}_idx": dim_to_idx,
            'hotel_to_idx': hotel_to_idx,
            'criteria': self.criteria
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model đã được lưu vào {filename}")

    def load_model(self, filename='model.pkl'):
        """Tải model từ file."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} không tồn tại")
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        dim_idx_key = 'country_idx' if 'country_idx' in model_data else 'status_to_idx'
        return (model_data['weights'], model_data['factors'],
                model_data[dim_idx_key], model_data['hotel_to_idx'], model_data['criteria'])

    def predict_with_saved_model(self, dim_value, hotel, model_filename='model.pkl', dimension='country'):
        """
        Dự đoán điểm số sử dụng model đã lưu.
        
        Args:
            dim_value: Giá trị của dimension (quốc gia hoặc trạng thái)
            hotel: ID khách sạn
            model_filename: Đường dẫn file model
            dimension: 'country' hoặc 'state'
        """
        weights, factors, dim_to_idx, hotel_to_idx, criteria = self.load_model(model_filename)
        
        if dim_value not in dim_to_idx:
            raise ValueError(f"{dimension.capitalize()} '{dim_value}' không có trong dữ liệu")
        if hotel not in hotel_to_idx:
            raise ValueError(f"Khách sạn '{hotel}' không có trong dữ liệu")
        
        dim_idx = dim_to_idx[dim_value]
        hotel_idx = hotel_to_idx[hotel]
        predicted_scores = self.predict_scores(dim_idx, hotel_idx, factors, weights)
        
        print(f"\nDự đoán điểm số cho {dimension} '{dim_value}', khách sạn '{hotel}':")
        for crit, score in zip(criteria, predicted_scores):
            print(f"{crit}: {score:.3f}")
        
        return predicted_scores

    def aggregate_prediction(self, hotel_ids: list, country: str = None, status: str = None, result_groupby_id = None,
                              state_model_file='CF_state_for_user.pkl', 
                              country_model_file='CF_country_for_user.pkl', 
                              state_model=None, 
                              country_model=None, 
                              print_result=True, 
                              print_warning=True):

        """
        Dự đoán điểm số cho danh sách khách sạn dựa trên quốc gia, trạng thái người dùng hoặc dữ liệu tổng hợp.
        
        Args:
            hotel_ids (list): Danh sách ID khách sạn.
            country (str or None): Quốc gia người dùng.
            status (str or None): Trạng thái người dùng.
            result_groupby_id (dict): Dictionary chứa điểm số tổng hợp theo ID khách sạn.
            state_model_file (str): Đường dẫn file mô hình trạng thái.
            country_model_file (str): Đường dẫn file mô hình quốc gia.
            state_model (tuple): Tuple chứa mô hình trạng thái (weights, factors, dim_to_idx, hotel_to_idx, criteria).
            country_model (tuple): Tuple chứa mô hình quốc gia (weights, factors, dim_to_idx, hotel_to_idx, criteria).
            print_result (bool): Có in kết quả hay không.
        
        Returns:
            list: Danh sách điểm số (vector 16 chiều) cho từng khách sạn.
        """
        # Khởi tạo kết quả
        results = []
        
        # Trường hợp cả country và status đều None: Dùng dữ liệu tổng hợp
        if country is None and status is None:
            for hotel in hotel_ids:
                try:
                    row = result_groupby_id[result_groupby_id['id'] == int(hotel)]
                    scores = row.drop(columns='id').values.flatten().tolist()
                    if len(scores) != len(self.criteria):
                        if print_warning:
                            print(f"Cảnh báo: Điểm tổng hợp của khách sạn '{hotel}' không đúng kích thước, trả về vector 0")
                        scores = np.zeros(len(self.criteria))
                except:
                    if print_warning:
                        print(f"Cảnh báo: Khách sạn '{hotel}' không có trong dữ liệu tổng hợp, trả về vector 0")
                    scores = np.zeros(len(self.criteria))
                results.append(scores)
            
            if print_result:
                print("\nDự đoán điểm số từ dữ liệu tổng hợp:")
                for hotel, scores in zip(hotel_ids, results):
                    print(f"\nKhách sạn '{hotel}':")
                    for crit, score in zip(self.criteria, scores):
                        print(f"  {crit}: {score:.3f}")
            
            return results
       
        if state_model is not None and country_model is not None:
            state_weights, state_factors, state_to_idx, state_hotel_to_idx, state_criteria = state_model
            country_weights, country_factors, country_to_idx, country_hotel_to_idx, country_criteria = country_model
        else:
            state_weights, state_factors, state_to_idx, state_hotel_to_idx, state_criteria = self.load_model(state_model_file)
            country_weights, country_factors, country_to_idx, country_hotel_to_idx, country_criteria = self.load_model(country_model_file)

        if country is not None and status is None:
            # Chỉ dự đoán theo quốc gia
            if country not in country_to_idx:
                raise ValueError(f"Quốc gia '{country}' không có trong mô hình quốc gia")
            
            country_idx = country_to_idx[country]
            for hotel in hotel_ids:
                if hotel not in country_hotel_to_idx:
                    if print_warning:
                        print(f"Cảnh báo: Khách sạn '{hotel}' không có trong dữ liệu, bỏ qua")
                    results.append(np.zeros(len(self.criteria)))
                    continue
                
                country_hotel_idx = country_hotel_to_idx[hotel]
                scores = self.predict_scores(country_idx, country_hotel_idx, country_factors, country_weights)
                results.append(scores)
            
            if print_result:
                print(f"\nDự đoán điểm số cho quốc gia '{country}':")
                for hotel, scores in zip(hotel_ids, results):
                    print(f"\nKhách sạn '{hotel}':")
                    for crit, score in zip(self.criteria, scores):
                        print(f"  {crit}: {score:.3f}")
            
            return results
            

        elif status is not None and country is None:
            # Chỉ dự đoán theo trạng thái
            if status not in state_to_idx:
                raise ValueError(f"Trạng thái '{status}' không có trong mô hình trạng thái")
            
            status_idx = state_to_idx[status]
            for hotel in hotel_ids:
                if hotel not in state_hotel_to_idx:
                    if print_warning:
                        print(f"Cảnh báo: Khách sạn '{hotel}' không có trong dữ liệu, bỏ qua")
                    results.append(np.zeros(len(self.criteria)))
                    continue
                
                state_hotel_idx = state_hotel_to_idx[hotel]
                scores = self.predict_scores(status_idx, state_hotel_idx, state_factors, state_weights)
                results.append(scores)

            if print_result:
                print(f"\nDự đoán điểm số cho trạng thái '{status}':")
                for hotel, scores in zip(hotel_ids, results):
                    print(f"\nKhách sạn '{hotel}':")
                    for crit, score in zip(self.criteria, scores):
                        print(f"  {crit}: {score:.3f}")
            
            return results

        if state_criteria != country_criteria or state_criteria != self.criteria:
            raise ValueError("Danh sách tiêu chí giữa hai mô hình và code không khớp")
        
        if status not in state_to_idx:
            raise ValueError(f"Trạng thái '{status}' không có trong mô hình trạng thái")
        if country not in country_to_idx:
            raise ValueError(f"Quốc gia '{country}' không có trong mô hình quốc gia")
        

        for hotel in hotel_ids:
            if hotel not in state_hotel_to_idx or hotel not in country_hotel_to_idx:
                if print_warning:
                    print(f"Cảnh báo: Khách sạn '{hotel}' không có trong dữ liệu, bỏ qua")
                results.append(np.zeros(len(self.criteria)))
                continue
            
            status_idx = state_to_idx[status]
            country_idx = country_to_idx[country]
            state_hotel_idx = state_hotel_to_idx[hotel]
            country_hotel_idx = country_hotel_to_idx[hotel]
            
            state_scores = self.predict_scores(status_idx, state_hotel_idx, state_factors, state_weights)
            country_scores = self.predict_scores(country_idx, country_hotel_idx, country_factors, country_weights)
            
            combined_scores = (state_scores + country_scores) / 2
            results.append(combined_scores)
        
        if print_result:
            print(f"\nDự đoán điểm số cho trạng thái '{status}', quốc gia '{country}':")
            for hotel, scores in zip(hotel_ids, results):
                print(f"\nKhách sạn '{hotel}':")
                for crit, score in zip(self.criteria, scores):
                    print(f"  {crit}: {score:.3f}")

        results_ = {}
        for id, scores in zip(hotel_ids, results):
            results_[id] = {}
            for crit, score in zip(self.criteria, scores):
                results_[id][crit.replace('_Positive', '')] = score
        
        return results_