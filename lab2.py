import cv2
import numpy as np
import matplotlib.pyplot as plt


def threshold_change(th):
    global final
    final = cv2.threshold(summed, th, 255, cv2.THRESH_BINARY)[1]


lenna = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
threshold = 70

filters = np.empty((4, 3, 3), dtype=np.int8)
filters[0] = [[1, 1, 0],
              [1, 0, -1],
              [0, -1, -1]]
filters[1] = [[1, 1, 1],
              [0, 0, 0],
              [-1, -1, -1]]
filters[2] = [[0, 1, 1],
              [-1, 0, 1],
              [-1, -1, 0]]
filters[3] = [[-1, 0, 1],
              [-1, 0, 1],
              [-1, 0, 1]]
filters = np.append(filters, np.negative(filters), axis=0)

filtered = [cv2.filter2D(lenna, -1, fil) for fil in filters]
summed = np.max(filtered, axis=0)

thresholded = [cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
               for image in filtered]

plt.figure(1)
x = 241
for i in range(len(thresholded)):
    plt.subplot(x+i)
    plt.imshow(thresholded[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(str(filters[i]))
plt.suptitle('Compass operators\nThreshold: {}'.format(threshold))
plt.show()

final = cv2.threshold(summed, threshold, 255, cv2.THRESH_BINARY)[1]

cv2.namedWindow('image')
cv2.createTrackbar('Threshold', 'image', 0, 255, threshold_change)
cv2.setTrackbarPos('Threshold', 'image', threshold)
while(1):
    cv2.imshow('image', final)
    k = cv2.waitKey(1)
    if k is ord('q'):
        break
cv2.destroyAllWindows()
