import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

lookup = np.empty(shape=256, dtype=np.uint8)
points = [(0, 0), (256, 256)]


def update_lookup(x0, y0, x1, y1):
    k = (y1 - y0) / (x1 - x0)
    for i in range(x0, x1):
        lookup[i] = math.floor((k * (i - x0) + y0))


def contrast_stretching(image_path, coordinates):
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    points[1:1] = coordinates

    for i in range(len(points)-1):
        update_lookup(*points[i], *points[i+1])

    stretched = [lookup[x] for x in original]

    plot(original, stretched, coordinates)


def plot(im1, im2, coords):
    fig = plt.figure()

    plt.subplot(131)
    plt.imshow(im1, cmap='gray')
    plt.title('Original image')

    plt.subplot(132)
    plt.imshow(im2, cmap='gray')
    plt.title('Contrast stretched image')

    ax = fig.add_subplot(1, 3, 3, aspect=1)
    ax.plot(lookup)
    if len(coords):
        xs, ys = zip(*coords)
        ax.plot(xs, ys, 'ro')

    plt.show()


if __name__ == "__main__":
    contrast_stretching('lena.png', [(60, 20), (150, 230)])
