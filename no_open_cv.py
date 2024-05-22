import cv2
import numpy as np
import math
from scipy.signal import convolve2d
import sys

sys.setrecursionlimit(100000)


def resizing(frame):
    width = int(frame.shape[1] * 50 / 100)
    height = int(frame.shape[0] * 50 / 100)
    dim = (width, height)
    return cv2.resize(frame, dim)


cap = cv2.VideoCapture('videos/snow.mp4')


def gauss(x, y, sigma):
    return 1 / (2 * math.pi * sigma ** 2) * math.exp((-x ** 2 - y ** 2) / (2 * sigma ** 2))


def Gauss_Filter(frame):
    sigma = 0.66

    k = round(6 * sigma) + 1
    kernel = np.array([[gauss(i - k // 2, j - k // 2, sigma) for j in range(k)] for i in range(k)])

    return convolve2d(frame, kernel / kernel.sum(), mode='valid').astype('uint8')


def dilate(frame_first, iter=1):
    img = frame_first
    while iter > 0:
        white_pics = np.where(img == 255)

        for x, y in zip(white_pics[0], white_pics[1]):
            img[x - 2:x + 1, y - 2:y + 1] = 255
        iter -= 1

    return img


def inRange(frame, lower, upper):
    return np.where(frame < lower, 0, 255).astype(np.uint8)


def threshold(frame, lower, max_val):
    return np.where(frame < lower, 0, max_val).astype(np.uint8)



def draw_rect(frame, cnt):
    x_min = min(cnt, key=lambda x: x[0])[0]
    x_max = max(cnt, key=lambda x: x[0])[0]
    y_min = min(cnt, key=lambda x: x[1])[1]
    y_max = max(cnt, key=lambda x: x[1])[1]
    if (x_max - x_min) * (y_max - y_min) > 1500:
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)


def find_contours(image):
    def get_neighbors(x, y):
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    height, weight = image.shape
    labeled_image = np.zeros((height, weight))
    label = 1

    white_pics = np.where(image == 255)

    for i, j in zip(white_pics[0], white_pics[1]):
        if labeled_image[i, j] == 0:
            stack = [(i, j)]
            while stack:
                x, y = stack.pop()
                if x < 0 or y < 0 or x >= height or y >= weight or labeled_image[x, y] != 0 or image[x, y] == 0:
                    continue
                labeled_image[x, y] = label
                for nx, ny in get_neighbors(x, y):
                    stack.append((nx, ny))
            label += 1

    contours = []
    for l in range(1, label):
        bac = np.where(labeled_image == l)
        cab = (bac[1], bac[0])
        contour = np.array(cab).T
        contours.append(contour)

    return contours


suc, frame = cap.read()
suc1, frame_next = cap.read()
frame = resizing(frame)
frame_next = resizing(frame_next)

while True:
    subtract = frame.astype(np.float32) - frame_next.astype(np.float32)
    my_diff = inRange(np.abs(subtract), (50, 50, 50), (255, 255, 255))

    gray = cv2.cvtColor(my_diff, cv2.COLOR_BGR2GRAY)

    blur = Gauss_Filter(gray)
    mask_obj = threshold(blur, 20, 255)
    dilated = dilate(mask_obj, 5)

    cnt = find_contours(dilated)

    for cont in cnt:
        draw_rect(frame, cont)

    #cv2.drawContours(frame, cnt, -1, (0, 255, 0), 2)

    cv2.imshow('diff', frame)
    #cv2.imshow('dil1', dilated)
    frame = frame_next  #
    ret, frame_next = cap.read()  #
    frame_next = resizing(frame_next)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
