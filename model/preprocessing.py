import cv2
import numpy as np
import model.coin_image as ci


def normalize(x_data: np.ndarray):
    tmp = x_data.astype(np.float64)
    tmp /= 255.0  # scale between 0 and 1
    """
    std = np.std(tmp, axis=(0, 1, 2))  # Global RGB std
    mean = np.mean(tmp, axis=(0, 1, 2))  # Global RGB mean
    print('Global RGB mean for dataset', mean)
    print('Global RGB STD for dataset', std)
    return (tmp - mean) / std
    """
    return tmp


def apply_gaussian_filter(img):
    gaussian = np.array([[1 / 16., 1 / 8., 1 / 16.], [1 / 8., 1 / 4., 1 / 8.], [1 / 16., 1 / 8., 1 / 16.]])
    dst = cv2.filter2D(img, -1, gaussian)
    return dst


def hough_transform(img, width, height):
    ret, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, dp=1, minDist=10, maxRadius=50, param1=50, param2=30,
                               minRadius=10)
    circles = filter_found_cicles((np.around(circles))[0])
    coins = []
    coin_images = []
    for circle in circles:
        x = int(circle[0] - circle[2])
        y = int(circle[1] - circle[2])
        local_height = int(2 * circle[2])
        local_width = int(2 * circle[2])
        coin_found = img[y:y + local_height, x:x + local_width]
        coins.append(cv2.resize(coin_found, (width, height)))
        coin_images.append(ci.CoinImage(int(circle[2])))
    return coins, coin_images


def filter_found_cicles(circles):
    ordered_circles = sorted(circles, key=lambda x: x[2])
    filtered_circles = []
    for i in range(len(ordered_circles) - 1, 0, -1):
        is_contained = False
        for circle in filtered_circles:
            if distance(circle, ordered_circles[i]) < circle[2]:
                is_contained = True
                break
        if not is_contained:
            filtered_circles.append(ordered_circles[i])
    return filtered_circles


def distance(circle1, circle2):
    dif_x = (circle1[0] - circle2[0])**2
    dif_y = (circle1[1] - circle2[1])**2
    return (dif_x + dif_y)**(1/2.0)


