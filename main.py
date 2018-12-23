import cv2
import numpy as np
from copy import deepcopy
import argparse


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Circle extractor')
    parser.add_argument('--save', action='store_true',
                        help='Save scaled images')
    args = vars(parser.parse_args())
    print('Args:', args)
    img = cv2.imread('data/coins.jpg', 0)
    cimg = cv2.imread('data/coins.jpg', cv2.IMREAD_COLOR)
    cimg2 = deepcopy(cimg)
    cv2.imshow('color image', cimg)
    cv2.waitKey(0)
    ret, dst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, dp=1, minDist=10, maxRadius=50, param1=50, param2=30, minRadius=10)
    circles = np.uint16(np.around(circles))
    j = 0
    for i in circles[0, :]:
        x = i[0] - i[2]
        y = i[1] - i[2]
        height = 2 * i[2]
        width = 2 * i[2]
        roi = cimg[y - 3:y + height + 6, x - 3:x + width + 6]
        new_roi = cv2.resize(roi, (100, 100))
        if args['save']:
            cv2.imwrite(str(j) + '.jpg', new_roi)
        else:
            pass
            # Uncomment to show scaled coin
            # cv2.imshow("Scaled coin", new_roi)
            # cv2.waitKey(0)

        # draw the outer circle
        cv2.circle(cimg2, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg2, (i[0], i[1]), 2, (0, 0, 255), 3)
        j += 1

    cv2.imshow('detected circles', cimg2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
