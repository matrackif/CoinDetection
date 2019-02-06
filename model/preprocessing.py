import cv2
import numpy as np
import model.coin_image as ci
from copy import deepcopy


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


def gaussian_smoothing(input_img):
    gaussian_filter = np.array([[0.109, 0.111, 0.109],
                                [0.111, 0.135, 0.111],
                                [0.109, 0.111, 0.109]])

    return cv2.filter2D(input_img, -1, gaussian_filter)


def canny_edge_detection(input):
    input = input.astype('uint8')

    # Using OTSU thresholding - bimodal image
    otsu_threshold_val, ret_matrix = cv2.threshold(input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # lower_threshold = otsu_threshold_val * 0.8
    # upper_threshold = otsu_threshold_val * 1.7

    lower_threshold = otsu_threshold_val * 0.4
    upper_threshold = otsu_threshold_val * 1.3

    print(lower_threshold, upper_threshold)

    # print(lower_threshold,upper_threshold)
    edges = cv2.Canny(input, lower_threshold, upper_threshold)
    return edges


def HoughCircles(input, circles):
    rows = input.shape[0]
    cols = input.shape[1]

    # initializing the angles to be computed
    sinang = dict()
    cosang = dict()

    # initializing the angles
    for angle in range(0, 360):
        sinang[angle] = np.sin(angle * np.pi / 180)
        cosang[angle] = np.cos(angle * np.pi / 180)

        # initializing the different radius
    # For Test Image <----------------------------PLEASE SEE BEFORE RUNNING------------------------------->
    # radius = [i for i in range(10,70)]
    # For Generic Images
    length = int(rows / 2)
    radius = [i for i in range(5, length)]

    # Initial threshold value
    threshold = 190

    for r in radius:
        # Initializing an empty 2D array with zeroes
        acc_cells = np.full((rows, cols), fill_value=0, dtype=np.uint64)

        # Iterating through the original image
        for x in range(rows):
            for y in range(cols):
                if input[x][y] == 255:  # edge
                    # increment in the accumulator cells
                    for angle in range(0, 360):
                        b = y - round(r * sinang[angle])
                        a = x - round(r * cosang[angle])
                        if a >= 0 and a < rows and b >= 0 and b < cols:
                            acc_cells[(int)(a)][(int)(b)] += 1

        print('For radius: ', r)
        acc_cell_max = np.amax(acc_cells)
        print('max acc value: ', acc_cell_max)

        if (acc_cell_max > 150):

            print("Detecting the circles for radius: ", r)

            # Initial threshold
            acc_cells[acc_cells < 150] = 0

            # find the circles for this radius
            for i in range(rows):
                for j in range(cols):
                    if (i > 0 and j > 0 and i < rows - 1 and j < cols - 1 and acc_cells[i][j] >= 150):
                        avg_sum = np.float32((acc_cells[i][j] + acc_cells[i - 1][j] + acc_cells[i + 1][j] +
                                              acc_cells[i][j - 1] + acc_cells[i][j + 1] + acc_cells[i - 1][j - 1] +
                                              acc_cells[i - 1][j + 1] + acc_cells[i + 1][j - 1] + acc_cells[i + 1][
                                                  j + 1]) / 9)
                        print("Intermediate avg_sum: ", avg_sum)
                        if (avg_sum >= 33):
                            print("For radius: ", r, "average: ", avg_sum, "\n")
                            circles.append((i, j, r))
                            acc_cells[i:i + 5, j:j + 7] = 0



img_path = '../data/images_of_multiple_coins/coins.jpg'
# sample_inp1_path = 'circle_sample_1.jpg'

orig_img = cv2.imread(img_path)

# Reading the input image and converting to gray scale
input = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Create copy of the orignial image
input_img = deepcopy(input)

# Steps
# 1. Denoise using Gaussian filter and detect edges using canny edge detector
smoothed_img = gaussian_smoothing(input_img)

# 2. Detect Edges using Canny Edge Detector
edged_image = canny_edge_detection(smoothed_img)
# 3. Detect Circle radius
# 4. Perform Circle Hough Transform
circles = []

# cv2.imshow('Circle Detected Image',edged_image)

# Detect Circle
HoughCircles(edged_image, circles)

# Print the output
for vertex in circles:
    cv2.circle(orig_img, (vertex[1], vertex[0]), vertex[2], (0, 255, 0), 1)
    cv2.rectangle(orig_img, (vertex[1] - 2, vertex[0] - 2), (vertex[1] - 2, vertex[0] - 2), (0, 0, 255), 3)

print(circles)

cv2.imshow('Circle Detected Image', orig_img)
cv2.imwrite('Circle_Detected_Image.jpg', orig_img)