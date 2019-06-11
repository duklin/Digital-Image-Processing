import cv2
import numpy as np
import glob
import pickle


def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response,
                point.octave, point.class_id, descriptors[i])
        i += 1
        temp_array.append(temp)
    return temp_array


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


if __name__ == "__main__":
    database_path = 'Database/'
    sift = cv2.xfeatures2d.SIFT_create()
    index = {}

    for image_path in glob.glob(database_path + '**jpg'):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        coef = min(1, 300 / img.shape[1])
        img = cv2.resize(img, None, fx=coef, fy=coef,
                         interpolation=cv2.INTER_AREA)
        (kps, descs) = sift.detectAndCompute(img, None)
        image_path = image_path.replace(database_path, '')
        index[image_path] = pickle_keypoints(kps, descs)

    with open('index.pickle', 'wb') as file:
        pickle.dump(index, file)
