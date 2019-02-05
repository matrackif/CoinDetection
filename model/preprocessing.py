import cv2
import numpy as np


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


"""TODO verification of algorithm"""
"""
source: https://github.com/SunilVasu/Circle-Hough-Transform/blob/master/HoughTransform_FOR_GIVEN_TEST_IMAGE.py
def detect_circles(img):
    circles = []
    rows = img.shape[0]
    cols = img.shape[1]

    # initializing the angles to be computed
    sinang = dict()
    cosang = dict()

    # initializing the angles
    for angle in range(0, 360):
        sinang[angle] = np.sin(angle * np.pi / 180)
        cosang[angle] = np.cos(angle * np.pi / 180)

        # initializing the different radius
    # For Given Test Image <----------------------PLEASE SEE BEFORE RUNNING------------------------------->
    radius = [i for i in range(10, 70)]
    # For Generic Images
    # length=int(rows/2)
    # radius = [i for i in range(5,length)]

    # Initial threshold value
    threshold = 190

    for r in radius:
        # Initializing an empty 2D array with zeroes
        acc_cells = np.full((rows, cols), fill_value=0, dtype=np.uint64)

        # Iterating through the original image
        for x in range(rows):
            for y in range(cols):
                if img[x][y] == 255:  # edge
                    # increment in the accumulator cells
                    for angle in range(0, 360):
                        b = y - round(r * sinang[angle])
                        a = x - round(r * cosang[angle])
                        if a >= 0 and a < rows and b >= 0 and b < cols:
                            acc_cells[a][b] += 1

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
    return circles
"""

