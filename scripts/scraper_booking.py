from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium.webdriver.support.ui import Select
import time
import requests
import re
from collections import defaultdict
from parsel import Selector
import pandas as pd
import json
from datetime import datetime


def wait_for_elements(driver, timeout=10):
    """
    chờ phần tử xuất hiện.
    """
    # get_nearby_places
    # Chờ section xuất hiện trước
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, "surroundings_block"))
    )

    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#surroundings_block div.a53cbfa6de.f45d8e4c32.cea0c192d7"))
    )

    """
    wait for hotel facility
    """
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='property-section--content'] div.e50d7535fa div.f1e6195c8b"))
    )
    

def click_for_expand_images(driver, timeout=10):
    """
    # click get images
    """

    button = driver.find_element(By.XPATH, '//div[@class="k2-hp--gallery-header bui-grid__column bui-grid__column-9"]')
    button_expand_image = button.find_element(By.CLASS_NAME, "a83ed08757.e03b6bd5da")

    # Cuộn đến phần tử
    driver.execute_script("arguments[0].scrollIntoView();", button_expand_image)

    # Click vào nút
    button_expand_image.click()
    
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CLASS_NAME, "bh-photo-modal-masonry-grid-item"))
    )

def get_stars_rating(driver):
    try:
        rating_element = driver.find_element(By.XPATH, "//span[@data-testid='rating-stars']")
    except:
        try:
            rating_element = driver.find_element(By.XPATH, "//span[@data-testid='rating-squares']")
        except:
            try:
                rating_element = driver.find_element(By.XPATH, "//div[@data-testid='quality-rating']")
            except:
                return None
    stars = rating_element.find_elements(By.TAG_NAME, "svg")
    return(len(stars))
def get_info_room_type(driver):
    result = {}

    # Lấy số lượng các loại phòng
    rooms = driver.find_elements(By.CLASS_NAME, 'hprt-roomtype-icon-link')
    num_room_types = len(rooms)
    result['num_roomtypes'] = num_room_types

    # get infor each room type
    tbody = driver.find_element(By.TAG_NAME, "tbody")
    rows = tbody.find_elements(By.TAG_NAME, "tr")
    list_roomtypes = []
    for row in rows:
        package_other = False
        info_roomtypes = {}

        try:
            # Lấy tên loại phòng
            room = row.find_element(By.CLASS_NAME, 'hprt-roomtype-icon-link')
            info_roomtypes['name'] = room.text
        except:
            info_roomtypes['name'] = "package_room_other"
            package_other = True
        


        # get price information
        price_elements = row.find_element(By.CLASS_NAME, "hprt-price-block")
        try:
            original_price = price_elements.find_element(By.CLASS_NAME, "bui-price-display__original").text.strip()
        except:
            original_price = None

        try:
            current_price = price_elements.find_element(By.CLASS_NAME, "bui-price-display__value").text.strip()
        except:
            current_price = None

        if original_price is None:
            original_price = current_price
        elif current_price is None:
            current_price = original_price

        info_roomtypes['original_price'] = original_price
        info_roomtypes['current_price'] = current_price

        # get capacity room
        try:
            capacity_element = row.find_element(By.CSS_SELECTOR, "td.hprt-table-cell.hprt-table-cell-occupancy.droom_seperator")
        except:
            capacity_element = row.find_element(By.CSS_SELECTOR, "td.hprt-table-cell.hprt-table-cell-occupancy")
        try:
            # Lấy số lượng khách nếu có multiplier
            num_guests = capacity_element.find_element(By.CLASS_NAME, 'c-occupancy-icons__multiplier-number').text
            info_roomtypes['capacity'] = num_guests

        except:
            guests_element = capacity_element.find_element(By.CSS_SELECTOR, "span.bui-u-sr-only")
            number_of_guests_text = guests_element.text
            info_roomtypes['capacity'] = number_of_guests_text

        # get id room
        if package_other == False:
            room_div = row.find_element(By.XPATH, ".//div[contains(@class, 'hprt-roomtype-block') and contains(@class, 'hprt-roomtype-name') and contains(@class, 'hp_rt_room_name_icon__container')]")
            room_link = room_div.find_element(By.XPATH, ".//a[@href]")
            room_href = room_link.get_attribute("href").split('#')[1]
            info_roomtypes['id'] = room_href

            driver.execute_script("var rect = arguments[0].getBoundingClientRect(); window.scrollBy(0, rect.top - (window.innerHeight / 2));", room_link)
            
            room_link.click()
            time.sleep(1)
            driver.page_source

            try:
                info_room = driver.find_element(By.XPATH, "//button[@type='button' and contains(@class, 'bui-modal__close') and @aria-label='Close dialog' and @data-bui-ref='modal-close']/following-sibling::*[1]")
            except:
                try:
                    info_room = driver.find_element(By.XPATH, "//button[@class='modal-mask-closeBtn' and @title='Đóng']/following-sibling::*[1]")
                except:
                    info_room = driver.find_element(By.CLASS_NAME, "f7c2c6294c")

            # get image for rooms
            images = []
            try:
                imgs = info_room.find_element(By.XPATH, ".//div[@class='b_nha_hotel_small_images']").find_elements(By.TAG_NAME, "img") 
                for img in imgs:
                    alt_text = img.get_attribute("alt")  # Lấy alt
                    img_url = img.get_attribute("src")  # Lấy src
                    images.append({"alt": alt_text, "url": img_url})
            except:
                imgs = info_room.find_element(By.XPATH, ".//div[@class='cd448175ab']").find_elements(By.TAG_NAME, "img") 
                for img in imgs:
                    alt_text = img.get_attribute("alt")  # Lấy alt
                    img_url = img.get_attribute("src")  # Lấy src
                    images.append({"alt": alt_text, "url": img_url})
            info_roomtypes['images'] = images

            # get amenities for room
            amenties = info_room.find_elements(By.XPATH, ".//div[@class='hprt-facilities-facility']")
            room_amenties = []
            for amentie in amenties:
                room_amenties.append(amentie.text)
            info_roomtypes['amenities'] = room_amenties
            try:
                if not room_amenties:
                    amenties = info_room.find_element(By.CLASS_NAME, "a42e82ebcd").text.split("\n")
                    for amentie in amenties:
                        room_amenties.append(amentie)
                    info_roomtypes['amenities'] = room_amenties
            except:
                pass

            # get bed types for room
            try:
                bed_types = info_room.find_element(By.XPATH, ".//span[@class='wholesalers_table__bed_options__text']").text
            except:
                try:
                    bed_types = info_room.find_element(By.XPATH, ".//div[@class='roomtype-no-margins']").text
                except:
                    try:
                        bed_types = info_room.find_element(By.XPATH, ".//span[@class='baf7cb1417']").text
                    except:
                        bed_types = row.find_element(By.CLASS_NAME, 'hprt-roomtype-bed')
                        beds = bed_types.find_elements(By.CLASS_NAME, 'rt-bed-types')
                        bed_types = " 【 】 ".join([element.text for element in beds])



            info_roomtypes['bed_types'] = bed_types

            # get facilities for room
            try:
                room_facilities = info_room.find_element(By.XPATH, ".//div[@class='more-facilities-space']").text
            except:
                room_facilities_ = info_room.find_elements(By.CLASS_NAME, "d8dbf887aa")[-3:]
                # Lấy text của mỗi phần tử và gộp lại bằng "\n"
                room_facilities = "\n".join([facility.text for facility in room_facilities_])
                
            lines = room_facilities.split("\n")  # Tách chuỗi thành danh sách các dòng

            room_facilitie = {}
            current_category = None

            for line in lines:
                line = line.strip()  # Loại bỏ khoảng trắng dư thừa
                if not line:
                    continue  # Bỏ qua dòng trống
                
                if line.endswith(":"):  # Nếu dòng là tiêu đề danh mục
                    current_category = line[:-1]  # Loại bỏ dấu `:`
                    room_facilitie[current_category] = []  # Tạo danh sách rỗng
                elif current_category:  # Nếu có danh mục hiện tại
                    room_facilitie[current_category].append(line)  # Thêm vào danh sách
            
            info_roomtypes['facilities'] = room_facilitie
            

            try:
                close_button = driver.find_element(By.XPATH, "//button[@type='button' and contains(@class, 'bui-modal__close') and @aria-label='Close dialog' and @data-bui-ref='modal-close']")
                close_button.click()
            except:
                try:
                    close_button = driver.find_element(By.XPATH, "//button[@class='modal-mask-closeBtn' and @title='Đóng']")
                    close_button.click()
                except:
                    close_button = driver.find_element(By.XPATH, "//button[@type='button' and @aria-label='' and contains(@class, 'a83ed08757') and contains(@class, 'c21c56c305') and contains(@class, 'f38b6daa18') and contains(@class, 'd691166b09') and contains(@class, 'ab98298258') and contains(@class, 'f4552b6561')]")
                    close_button.click()
                    
        # get service room
        room_service_included = row.find_elements(By.TAG_NAME, "td")[3]
        info_roomtypes['room_service_included'] = room_service_included.text.split("•")[0].strip()
    
        list_roomtypes.append(info_roomtypes)

    result['infomation_roomtypes'] = list_roomtypes
    return result

def get_policy(driver):
    # Find the section containing policies
    policies_hotels = {}
    policies_section = driver.find_element(By.ID, "hp_policies_box")
    policies = policies_section.find_elements(By.CLASS_NAME, "a26e4f0adb")
    for policy in policies:
        item = policy.find_element(By.CLASS_NAME, "c6e1dbf31b").text
        detail = policy.find_element(By.CLASS_NAME, "f565581f7e").text
        policies_hotels[item] = detail
    return policies_hotels
    
def get_ID_hotel(driver):
    html = driver.page_source
    return re.findall(r"b_hotel_id:\s*'(.+?)'", html)[0] if re.findall(r"b_hotel_id:\s*'(.+?)'", html) else ""
def get_info(driver, html: str, url, IS_GET_INFO_HOTEL = True,
                                   IS_GET_INFOMATION_ROOM = True,
                                   IS_GET_POLICY = True):
    sel = Selector(text=html)

    css = lambda selector, sep="": sep.join(sel.css(selector).getall()).strip()
    css_first = lambda selector: sel.css(selector).get("")


    # Lấy tọa độ khách sạn (lat, lng)
    lat_lng = css_first("a[data-atlas-latlng]::attr(data-atlas-latlng)")
    lat, lng = lat_lng.split(",") if lat_lng else ("", "")
    

    # Lấy điểm đánh giá khách sạn
    rate = css_first("div.a3b8729ab1.d86cee9b25::text")

    
    # Lấy số lượng đánh giá khách sạn
    try:
        review_text = css_first("div.abf093bdfe.f45d8e4c32.d935416c47::text")
        review_count = review_text
    except:
        review_count = None
    
    # Lấy địa chỉ khách sạn
    try:
        address = css_first("div[data-testid='PropertyHeaderAddressDesktop-wrapper'] span.f419a93f12 > div::text")
        address_text = address.strip() if address else None
    except:
        address_text = None

    if IS_GET_INFO_HOTEL:
        # Lấy các tiện ích khách sạn phổ biến
        popular_facilities = set()
        try:
            # Vị trí 1
            popular_facilities.update(
                sel.css("div[data-testid='property-highlights'] ul li div.a53cbfa6de.ebbf62ced0::text").getall()
            )
        except:
            pass
        
        try:
            # Vị trí 2
            popular_facilities.update(
                sel.css("div[data-testid='property-most-popular-facilities-wrapper'] ul li div.a53cbfa6de.e6208ee469 span.a5a5a75131::text").getall()
            )
        except:
            pass

        # Lấy mô tả
        description_parts = []
        try:
            description_parts.append(css_first("div.hp-description span.afad290af2::text"))
        except:
            pass
        try:
            description_parts.append(css_first("p[data-testid='property-description']::text"))
        except:
            pass
        try:
            location_score_html = css_first("p[data-testid='property-description-location-score-trans']")
            location_score_text = re.sub(r'<p[^>]*>|</p>', '', location_score_html)  # Xóa thẻ <p>
            description_parts.append(location_score_text)
        except:
            pass
        description = " 📖 ".join(filter(None, description_parts))  # Ghép lại, bỏ phần rỗng

        # Lấy các tiện ích chi tiết
        hotel_facilities = []
        try:
            sections = sel.css("div[data-testid='property-section--content'] div.e50d7535fa div.f1e6195c8b")  

            for section in sections:
                try:
                    title1 = section.css("div.e1eebb6a1e.e6208ee469.d0caee4251").xpath("string(.)").get(default=" ").strip()
                    title2 = section.css("div.a53cbfa6de.f45d8e4c32.df64fda51b").xpath("string(.)").get(default=" ").strip()
                    title = " ➜ ".join([title1, title2])
                    #print("===")
                except:
                    title = section.css("h3").xpath("string(.)").get(default=" ").strip()
                    

                items = [
                    " 📌 ".join(item.css("span.a5a5a75131 *::text").getall()).strip()
                    for item in section.css("ul.c807d72881.da08adf9d2.e10711a42e li.a8b57ad3ff.d50c412d31.fb9a5438f9.c7a5a1307a")
                ]

                hotel_facilities.append({"title": title, "items": items})
        except:
            hotel_facilities = None


        
        # Lấy tiện ích xung quanh
        poi_blocks = driver.find_elements(By.CSS_SELECTOR, "#surroundings_block .f1bc79b259 .d31796cb42[data-testid='poi-block']")
        nearby_places = []
        for block in poi_blocks:
            try:
                # Lấy tiêu đề (tên mục) từ <ul data-testid="poi-block-list">
                title_element = block.find_element(By.CLASS_NAME, "e1eebb6a1e.e6208ee469.d0caee4251")
                title = title_element.text.strip() if title_element else "Không có tên"
                
                places = block.find_elements(By.XPATH, './/ul[@data-testid="poi-block-list"]/li')
                list_place = {}
                for place in places:
                    name = place.find_element(By.CLASS_NAME, 'dc5041d860.c72df67c95.fb60b9836d')
                    span_element = name.find_elements(By.TAG_NAME, "span")
                    if span_element:
                        content = f"{span_element[0].text.strip()} ❂ {name.text.replace(span_element[0].text, '').strip()}"
                    else:
                        content = name.text.strip()
                        
                    distance = place.find_element(By.CLASS_NAME, 'a53cbfa6de.f45d8e4c32.cea0c192d7')
                    list_place[content] = distance.text.strip()
                # Lưu vào danh sách
                nearby_places.append({"title": title, "detail": list_place})
            except:
                pass

        sub_rating = {}
        evaluate = driver.find_element(By.XPATH, '//div[@data-testid="PropertyReviewsRegionBlock"]//div[@class="bui-spacer--larger"]')
        categorys = evaluate.find_elements(By.XPATH, './/div[@class="b817090550 a7cf1a6b1d"]')
        for category in categorys:
            sub_name, sub_rate = category.find_elements(By.XPATH, './/div[@class="c72df67c95"]')
            sub_rating[sub_name.text] = sub_rate.text
    else:
        popular_facilities = None
        hotel_facilities = None
        nearby_places = None
        sub_rating = None
        description_parts = None

    
    if IS_GET_INFOMATION_ROOM:
        infomation_roomtypes = get_info_room_type(driver)
    else:
        infomation_roomtypes = None
    
    if IS_GET_POLICY:
        policies_hotels = get_policy(driver)
    else:
        policies_hotels = None

    click_for_expand_images(driver, timeout=10)
    driver.page_source
    # Lấy hình ảnh khách sạn
    updated_images = driver.find_elements(By.CLASS_NAME, "bh-photo-modal-masonry-grid-item")
    images = []
    for item in updated_images:
        #print(item.get_attribute("outerHTML"))
        try:
            img = item.find_element(By.TAG_NAME, "img")  # Tìm thẻ <img>
            alt_text = img.get_attribute("alt")  # Lấy alt
            img_url = img.get_attribute("data-src")  # Lấy src
            images.append({"alt": alt_text, "url": img_url})
        except:
            pass

    result = {
        "title": css("div#hp_hotel_name h2::text"),
        "url": url,
        "lat_lng": (lat, lng),
        "rate": rate,
        "sub_rate": sub_rating,
        "review_count": review_count,
        "address": address_text,
        "popular_facilities": list(popular_facilities),
        "description": description,
        "hotel_facilities": hotel_facilities,
        "images": images,
        "nearby_places": nearby_places,
        "policies_hotels": policies_hotels,
        "info_roomtypes": infomation_roomtypes
    }
    return result


def is_review_in_past(review_date_text, compare_date):
    try:
        # Trích xuất ngày từ chuỗi (Ví dụ: "Ngày đánh giá: ngày 2 tháng 1 năm 2023")
        date_part = review_date_text.replace("Ngày đánh giá: ", "").strip()
        
        # Chuyển đổi chuỗi thành định dạng ngày tháng
        review_date = datetime.strptime(date_part, "ngày %d tháng %m năm %Y")
        
        # So sánh với ngày cho trước
        return review_date > compare_date
    
    except Exception as e:
        print(f"Lỗi khi xử lý ngày đánh giá: {e}")
        return True  # Nếu lỗi, mặc định bỏ qua
def get_review(url, driver):
    driver.get(url + "#tab-reviews")
    time.sleep(2)
    driver.execute_script("document.body.style.zoom='50%'")
    #select_element = driver.find_element(By.XPATH, "//select[@data-testid='reviews-sorter-component']")
    #select = Select(select_element)
    #select.select_by_value("NEWEST_FIRST")
    result = []
    while True:
        should_break = False

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "dd5dccd82f"))
        )
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "c78681b5ef"))
        )
        #driver.page_source
        
        review_section = driver.find_element(By.CLASS_NAME, "dd5dccd82f").find_element(By.CLASS_NAME, "c78681b5ef")
        reviews = review_section.find_elements(By.XPATH, ".//div[@data-testid='review-card']")
        navigation_page = review_section.find_element(By.CSS_SELECTOR, "div[role='navigation'].c82435a4b8.a178069f51.a6ae3c2b40.a18aeea94d.d794b7a0f7.f53e278e95.e49b423746")
        for review in reviews:
            customers = {}
            try:
                user_reviews= review.find_element(By.XPATH, ".//div[@role='group' and @aria-label='Người viết đánh giá' and @class='b817090550 c44c37515e']")
                review_avatar = user_reviews.find_element(By.XPATH, ".//div[@data-testid='review-avatar']")
                user_reviews_info = user_reviews.find_element(By.XPATH, ".//div[@data-testid='review-stay-info']").find_elements(By.XPATH, ".//li")
                
                review_avatar_name = review_avatar.find_element(By.XPATH, ".//div[@class='a3332d346a e6208ee469']").text
                review_avatar_contry = review_avatar.find_element(By.XPATH, ".//span[@class='afac1f68d9 a1ad95c055']").text

                customers['name'] = review_avatar_name
                customers['country'] = review_avatar_contry
                customers['room'] = user_reviews_info[0].text
                customers['date'] = user_reviews_info[1].text
                customers['state'] = user_reviews_info[2].text
            except:
                pass
            
            review_content = review.find_element(By.XPATH, ".//div[@role='group' and @aria-label='Nội dung đánh giá' and @class='b817090550 d6cb5ce5de']")

            review_title = review_content.find_element(By.CSS_SELECTOR, "div.c624d7469d.f034cf5568.a8a3d245a8.a3214e5942.db150fece4.ecc4cec182")
            review_score = review_content.find_element(By.CSS_SELECTOR, "div.c624d7469d.f034cf5568.a8a3d245a8.a3214e5942.db150fece4.ecc4cec182").find_element(By.XPATH, ".//div[@data-testid='review-score']//div[@class='ac4a7896c7']").text

            try:
                review_date = review_title.find_element(By.XPATH, ".//span[@data-testid='review-date']").text
            except:
                review_date = None

            compare_date = datetime(2024, 1, 1)  # Ví dụ: Chỉ lấy các đánh giá từ ngày 01/01/2024 trở đi

            try:
                review_title = review_title.find_element(By.XPATH, ".//h3").text
            except:
                review_title = None
            try:
                review_score = review_content.find_element(By.CSS_SELECTOR, "div.c624d7469d.f034cf5568.a8a3d245a8.a3214e5942.db150fece4.ecc4cec182").find_element(By.XPATH, ".//div[@data-testid='review-score']//div[@class='ac4a7896c7']").text
            except:
                review_score = None

            try:
                review_positive = review_content.find_element(By.XPATH, ".//div[@data-testid='review-positive-text']").text
            except:
                review_positive = None

            try:
                review_negative = review_content.find_element(By.XPATH, ".//div[@data-testid='review-negative-text']").text
            except:
                review_negative = None

            try:
                photo = {}
                review_photos = review_content.find_element(By.XPATH, ".//div[@data-testid='review-photos']").find_elements(By.TAG_NAME, "img")
                for review_photo in review_photos:
                    #print(review_photo.get_attribute("href"))
                    photo[review_photo.get_attribute("src")] = review_photo.get_attribute("alt")
            except:
                photo = None

            customers['review_title'] = review_title
            customers['review_date'] = review_date
            customers['review_score'] = review_score
            customers['review_positive'] = review_positive
            customers['review_negative'] = review_negative
            customers['review_photo'] = photo

            result.append(customers)
        try:
            button = navigation_page.find_element(By.XPATH, "//div[@class='b16a89683f cab1524053']//button")
            ActionChains(driver).move_to_element(button).perform()
            button.click()
        except (NoSuchElementException, ElementClickInterceptedException):
            break
    return result

def scrape_hotel_data(url, IS_GET_INFO_HOTEL = True,
                                   IS_GET_INFOMATION_ROOM = True,
                                   IS_GET_POLICY = True, 
                                   IS_GET_REVIEW = True):
    
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")  # Không tải ảnh
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    time.sleep(2)
    driver.execute_script("document.body.style.zoom='50%'")
    try:
        wait_for_elements(driver, timeout=10)
    except:
        pass
    html = driver.page_source

    hotel_data = {}
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hotel_data['time_crawl'] = current_time
    try:
        # Get hotel ID
        ID = get_ID_hotel(driver)
        hotel_data["id"] = ID
    except:
        pass
    try:
        # Get stars rating
        stars_rating = get_stars_rating(driver)
        hotel_data["stars_rating"] = stars_rating
    except:
        pass

    try: 
        # Get hotel information and nearby places, room types, and policies
        if IS_GET_INFO_HOTEL and IS_GET_INFOMATION_ROOM and IS_GET_POLICY:
            hotel_data['info'] = get_info(driver, html, url, 
                                    IS_GET_INFO_HOTEL,
                                    IS_GET_INFOMATION_ROOM,
                                    IS_GET_POLICY)
    except:
        IS_GET_INFO_HOTEL, IS_GET_INFOMATION_ROOM, IS_GET_POLICY = False, False, False
    
    try:
        if IS_GET_REVIEW:
            time.sleep(2)
            hotel_data["reviews"] = get_review(url, driver)
    except:
        IS_GET_REVIEW = False

    driver.quit()
    
    return hotel_data, IS_GET_INFO_HOTEL, IS_GET_REVIEW


def generate_booking_url(base_url, checkin_date, checkout_date):
    params = {
        "checkin": checkin_date,
        "checkout": checkout_date
    }
    
    query_string = "&".join([f"{key}={value}" for key, value in params.items()])
    full_url = f"{base_url}?{query_string}"
    
    return full_url

def run(url, IS_GET_INFO_HOTEL, IS_GET_REVIEW):
    CHECKIN = "2025-05-25"
    CHECKOUT = "2025-05-26"

    IS_GET_INFOMATION_ROOM = True
    IS_GET_POLICY = True


    url_temp = generate_booking_url(url, CHECKIN, CHECKOUT)
    data, _INFO, _REVIEW = scrape_hotel_data(url_temp, 
                                IS_GET_INFO_HOTEL,
                                IS_GET_INFOMATION_ROOM,
                                IS_GET_POLICY, 
                                IS_GET_REVIEW)
    name = url.split("hotel/vn/")[1].split(".")[0]
    try:
        ID = data['id']
    except:
        ID = 0000000

    file_name = f"hotel_data-{name}_ID{ID}_INFO-{_INFO}_ROOM-{IS_GET_INFOMATION_ROOM}_POLICY-{IS_GET_POLICY}_REVIEW-{_REVIEW}.json"
    
    with open(file_name, mode='a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")  # Xuống dòng để tách các JSON object

    print(f"Đã lưu dữ liệu cho {url}")
        
    