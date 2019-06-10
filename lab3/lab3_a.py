import cv2
import glob
import numpy as np
import os


def segment(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    ret3, th3 = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, se)
    final = cv2.morphologyEx(closed, cv2.MORPH_OPEN, se)

    return final


def save_img(src_path, original, result):
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    final = np.hstack((original, result))
    dest = destination_path + src_path.replace('/', '-')
    cv2.imwrite(dest, final)


source_paths = ['database/', 'query/']
destination_path = 'segmented/'

if not os.path.exists(destination_path):
    os.mkdir(destination_path)

for path in source_paths:
    for image_path in glob.glob(path + '**jpg', recursive=True):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        res = segment(img)
        save_img(image_path, img, res)
