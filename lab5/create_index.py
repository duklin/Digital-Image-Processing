import cv2
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
