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


def hough_transform(img_file: str, greyscale: bool = False):
    if greyscale:
        output_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    else:
        output_img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    showdebug("Original image", output_img)
    mask = coin_color_mask(output_img)
    showdebug("Merged mask", mask)
    only_coins = cv2.bitwise_and(output_img,output_img,mask=mask)
    showdebug("Image after color detection", only_coins)
    cv2.waitKey(0)

    grayscale_only_coins = cv2.cvtColor(only_coins, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        grayscale_only_coins, cv2.HOUGH_GRADIENT,
        dp=1, minDist=10, minRadius=10, maxRadius=300,
        param1=50, param2=30)
    circles = filter_found_cicles((np.around(circles))[0])
    # coins = []
    coin_images = []
    for circle in circles:
        x = int(circle[0] - circle[2])
        y = int(circle[1] - circle[2])
        local_height = int(2 * circle[2])
        local_width = int(2 * circle[2])
        coin_found = output_img[y:y + local_height, x:x + local_width]
        # cv2.imshow('lol', coin_found)
        # cv2.waitKey(0)
        # coins.append(cv2.resize(coin_found, (width, height)))
        coin_images.append(ci.CoinImage(radius=int(circle[2]), img_arr=coin_found))
    return coin_images


def coin_color_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detects the yellow and brown colors in coins.
    coin_yellow_mask = cv2.inRange(
        hsv,
        np.array([int(30/2), int(0.25*255), 0]),
        np.array([int(60/2), 255, 255]))
    showdebug("coin yellow mask", coin_yellow_mask)

    # Detects the gray color in coins.
    coin_gray_mask = cv2.inRange(
        hsv,
        np.array([int(15/2), int(0.45*255), 0]),
        np.array([int(30/2), 255, 255]))
    showdebug("coin gray mask", coin_gray_mask)

    # Detects the gray color in the ideal coin image.
    coin_ideal_gray_mask = cv2.inRange(
        hsv,
        np.array([int(45/2), 0, 0]),
        np.array([int(170/2), 255, 255]))
    showdebug("coin ideal gray mask", coin_ideal_gray_mask)

    # Detects the almost-white gray color in the ideal coin image.
    coin_ideal_bright_gray_mask = cv2.inRange(
        hsv,
        np.array([int(20/2), int(0.03*255), 0]),
        np.array([int(65/2), int(0.05*255), 255]))
    showdebug("coin ideal bright gray mask", coin_ideal_bright_gray_mask)

    mask = (
        coin_yellow_mask +
        coin_gray_mask +
        coin_ideal_gray_mask +
        coin_ideal_bright_gray_mask)

    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    # showdebug("Opened and dilated mask", mask)

    return mask

def showdebug(description: str, image):
    cv2.imshow(description, cv2.resize(image, (800, 800)))


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


