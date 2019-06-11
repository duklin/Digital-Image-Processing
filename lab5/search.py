import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt


def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1],
                                    _size=point[1], _angle=point[2],
                                    _response=point[3], _octave=point[4],
                                    _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)


def flann_knn_matcher(desc1, desc2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])
    return good


def resize_img(original, new_col_size):
    coef = min(1, new_col_size / original.shape[1])
    return cv2.resize(original, None, fx=coef, fy=coef,
                      interpolation=cv2.INTER_AREA)


def search(img_path, index):
    # load query image and resize
    query_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    query_img = resize_img(query_img, 500)
    query_img_gray = resize_img(query_img_gray, 500)

    # keypoints and descriptors for query image
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(query_img_gray, None)

    MIN_MATCH_COUNT = 10  # number of minimum good matching points
    result = {}
    max_inliers = 0
    for (k, v) in index.items():
        train_kps, train_desc = v
        good = flann_knn_matcher(descs, train_desc)
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32(
                [kps[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [train_kps[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            inliers_num = len([i for i in matchesMask if i == 1])
            if inliers_num > max_inliers:
                max_inliers = inliers_num
                result['img_path'] = k
                result['key_points'] = train_kps
                result['good_points'] = good
                result['mask'] = matchesMask

    for i in range(len(result['mask'])):
        if result['mask'][i] == 1:
            result['mask'][i] = [1, 0]
        else:
            result['mask'][i] = [0, 0]

    # show only images
    result_img = cv2.imread('Database/' + result['img_path'], cv2.IMREAD_COLOR)
    result_img = resize_img(result_img, 300)
    img3 = cv2.drawMatches(query_img, None, result_img, None, None, None)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3), plt.xticks([]), plt.yticks([])
    plt.show()

    # show images and SIFT descriptors
    query_img_sift = cv2.drawKeypoints(
        query_img, kps, None, color=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    result_img_sift = cv2.drawKeypoints(
        result_img, result['key_points'], None, color=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img3 = cv2.drawMatches(query_img_sift, None,
                           result_img_sift, None, None, None)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3), plt.xticks([]), plt.yticks([])
    plt.show()

    # show knn matching points
    img3 = cv2.drawMatchesKnn(
        query_img, kps, result_img, result['key_points'],
        result['good_points'], None, matchColor=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3)
    plt.show()

    # show only inliers
    draw_params = dict(matchColor=(0, 255, 255),
                       singlePointColor=None, matchesMask=result['mask'],
                       flags=2)
    img3 = cv2.drawMatchesKnn(
        query_img, kps, result_img, result['key_points'],
        result['good_points'], None, **draw_params)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3)
    plt.show()


if __name__ == "__main__":
    index = pickle.load(open('index.pickle', 'rb'))
    for (k, v) in index.items():
        (kps, descs) = unpickle_keypoints(v)
        index[k] = (kps, descs)
    query_img_path = 'Query/hw7_poster_2.jpg'
    search(query_img_path, index)
