import cv2
import mahotas
import numpy as np
import pickle
import csv
import glob
import os


def zernike_moments(img):
    if os.getcwd().endswith('lab3'):
        from lab3_a import segment
    else:
        from lab3.lab3_a import segment
    thresh = segment(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    outline = np.zeros(img.shape, dtype="uint8")
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(outline, [cnts], -1, 255, -1)

    return mahotas.features.zernike_moments(outline, radius=max(img.shape)/2,
                                            degree=15)


if __name__ == '__main__':
    source_paths = ['database/', 'query/']
    index = {}

    for path in source_paths:
        for image_path in glob.glob(path + '**jpg', recursive=True):
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            res = zernike_moments(img)
            index[image_path] = res

    w = csv.writer(open('zernike_moments.csv', 'w'))
    for key, val in index.items():
        w.writerow([key, val])

    with open('index.pickle', 'wb') as file:
        pickle.dump(index, file)
