def estimate_distance(bbox, class_id):
    height_pixel = bbox[3] - bbox[1]
    if height_pixel <= 0:
        return -1

    real_heights = {
        0: 1.55,   # người
        1: 1.0,   # xe đạp
        2: 1.5,   # ô tô
        3: 1.2,   # xe máy
        5: 3.5,   # xe buýt
        7: 4.0    # xe tải
    }
    real_height = real_heights.get(class_id, 1.7)
    f = 800  
    distance_m = (f * real_height) / height_pixel
    return round(distance_m, 2)

