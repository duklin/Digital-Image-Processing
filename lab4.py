import cv2
import pickle
from lab3.lab3_b import zernike_moments
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt

index = pickle.load(open('lab3/index.pickle', 'rb'))


def search(query_img_path):
    query_img = cv2.imread(query_img_path, cv2.IMREAD_COLOR)
    moments = zernike_moments(query_img)
    results = {}
    for (k, v) in index.items():
        d = dist.correlation(v, moments)
        results[k] = d
    results = sorted([(v, k) for (k, v) in results.items()])
    return results[:6]


if __name__ == '__main__':
    query_img_path = 'lab3/query/14729.jpg'
    result = search(query_img_path)
    x = 231
    for i in range(6):
        plt.subplot(x+i)
        img_path = 'lab3/' + result[i][1]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb), plt.xticks([]), plt.yticks([])
        plt.xlabel('Distance: {0:.5f}'.format(result[i][0]))
        plt.title(img_path)
    plt.show()
