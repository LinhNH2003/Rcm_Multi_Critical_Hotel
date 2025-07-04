from collections import defaultdict
import numpy as np

def filter_hotel_by_star_rating(star_input, hotels):
    """
    Filters a list of hotels based on a given star rating input.

    Args:
        star_input (int or None): The star rating to filter by. If None, no filtering is applied.
        hotels (list): List of hotel dictionaries, each containing a 'stars_rating' key.

    Returns:
        list: A list of hotel dictionaries that have star ratings within a specified range 
              around the star_input. The range includes the input star rating and any adjacent 
              star ratings, ensuring some flexibility in the filtering.
    """

    if star_input == None:
        return hotels
    elif star_input == 5:
        stars = [star_input - 1, star_input]
    elif star_input == 1:
        stars = [star_input, star_input +1 ]
    else:
        stars = [star_input-1, star_input, star_input+1]

    filter_hotel = []
    for hotel in hotels:
        if hotel['stars_rating'] in stars:
            filter_hotel.append(hotel)

    return filter_hotel

def get_score_rating(user_input, hotels):
    """
    Calculates the rating scores for a list of hotels based only on star rating from user input.

    Args:
        user_input (dict): User preferences with only 'stars_rating'.
        hotels (list): List of hotel dictionaries containing information to be used for scoring.

    Returns:
        list: A list of dictionaries, each containing the hotel id and score_rating.

    The function compares hotel star ratings with the user's preference.
    """

    user_stars = user_input.get('stars_rating')
    results = []

    for hotel in hotels:
        hotel_stars = hotel.get('stars_rating')

        # Match: 1.0 nếu giống số sao, ngược lại giảm dần theo mức độ lệch
        if hotel_stars is not None and user_stars is not None:
            star_diff = abs(hotel_stars - user_stars)
            # Giả sử độ lệch 0 -> điểm 1.0, lệch 1 sao -> điểm 0.7, lệch 2 sao -> 0.4, lệch >=3 sao -> 0.1
            if star_diff == 0:
                score = 1.0
            elif star_diff == 1:
                score = 0.7
            elif star_diff == 2:
                score = 0.4
            else:
                score = 0.1
        else:
            score = 1  # Thiếu thông tin thì cho điểm thấp nhất

        results.append({
            'id': hotel.get('id'),
            'score_rating': round(score, 4)
        })

    return results


def parse_price_vnd(price_str):
    """
    Parses a price string in VND format and converts it to an integer.

    Args:
        price_str (str): The price string, potentially prefixed with "VND" and containing 
                         dots as thousand separators.

    Returns:
        int: The numeric value of the price without formatting. Returns 0 if the input is 
             None or cannot be parsed as an integer.
    """

    if not price_str:
        return 0
    # Loại bỏ tiền tố "VND", khoảng trắng và dấu chấm
    cleaned = price_str.replace('VND', '').replace('.', '').strip()
    try:
        return int(cleaned)
    except ValueError:
        return 0
    

def get_avg_room_score_per_hotel(room_hotels):
    """
    Calculates the average room price score for each hotel.

    Args:
        room_hotels (list): A list of dictionaries containing room information and price scores.

    Returns:
        list: A list of dictionaries containing the hotel ID and the average room price score.
    """
    hotel_scores = defaultdict(list)

    for room in room_hotels:
        hotel_id = room.get('id')

        score = room.get('room_score_price', 0)
        if score and score > 0:
            hotel_scores[hotel_id].append(score)

    # Tính trung bình cho từng khách sạn
    results = []
    for hotel_id, scores in hotel_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            results.append({
                'id': hotel_id,
                'hotel_score_price': round(avg_score, 4)
            })

    return results


def get_max_room_score_per_hotel(room_hotels):
    """
    Tính điểm giá cao nhất cho từng khách sạn.

    Args:
        room_hotels (list): Danh sách các phòng khách sạn.

    Returns:
        list: Danh sách các khách sạn kèm điểm giá cao nhất.
    """
    hotel_scores = defaultdict(list)

    for room in room_hotels:
        hotel_id = room.get('id')
        score = room.get('room_score_price', 0)

        if score and score > 0:
            hotel_scores[hotel_id].append(score)

    results = []
    for hotel_id, scores in hotel_scores.items():
        if scores:
            max_score = max(scores)
            results.append({
                'id': hotel_id,
                'hotel_score_price': round(max_score, 4)
            })

    return results


def get_price_score(user_input, room_hotels, beta_left=2, beta_right=10):

    """
    Tính điểm giá với giảm chậm bên trái và giảm nhanh bên phải.

    Args:
        price: có thể là số hoặc tuple (a, b) biểu thị khoảng giá.
        beta_left: kiểm soát tốc độ giảm khi giá thấp hơn ideal_price.
        beta_right: kiểm soát tốc độ giảm khi giá cao hơn ideal_price.
        user_input (dict): Thông tin người dùng, bao gồm khoảng giá mong muốn.
        room_hotels (list): Danh sách các phòng khách sạn.

    Returns:
        list: Danh sách các phòng kèm điểm giá.
    """
    user_input = user_input.get('price_range', (0, 0))  # Lấy khoảng giá từ input, mặc định là (0, 0)
    if isinstance(user_input, tuple):
        price = (user_input[0] + user_input[1]) / 2
    else:
        price = user_input

    results = []

    for hotel in room_hotels:
        hotel_price = parse_price_vnd(hotel.get('original_price'))
        if hotel_price is None or hotel_price == 0:
            continue
        deviation = (price - hotel_price) / hotel_price  
    
        if price <= hotel_price:
            score_price = np.exp(-beta_left * (deviation ** 2))  
        else:
            score_price = np.exp(-beta_right * (deviation ** 2))


        if score_price > 0:
            results.append({
                'id': hotel.get('id'),
                'room_id': hotel.get('room_id'),
                'room_score_price': score_price
            })

    return results


