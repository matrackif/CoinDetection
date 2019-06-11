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


def hough_transform(img_file: str, greyscale: bool = False):
    color_img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if greyscale:
        output_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    else:
        output_img = color_img
    showdebug("Original image", color_img)
    mask = coin_color_mask(color_img)
    showdebug("Merged mask", mask)
    only_coins = cv2.bitwise_and(color_img,color_img,mask=mask)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    coins_with_contours = only_coins.copy()
    cv2.drawContours(coins_with_contours, contours, -1, (0,0,255), 3)
    showdebug("Image after color detection + contours", coins_with_contours)
    cv2.waitKey(0)

    min_radius = 10
    coin_images = []
    for contour in contours:
        xs = contour[:, :, 0]
        ys = contour[:, :, 1]
        minx, maxx = np.min(xs), np.max(xs)
        miny, maxy = np.min(ys), np.max(ys)
        width, height = maxx - minx, maxy - miny
        radius = (width + height) / 4
        if radius < min_radius:
            continue
        coin_found = output_img[miny:maxy, minx:maxx]
        cv2.imshow('coin', coin_found)
        cv2.waitKey(0)
        # coins.append(cv2.resize(coin_found, (width, height)))
        coin_images.append(ci.CoinImage(radius=int(radius), img_arr=coin_found))
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
