import numpy as np
from sklearn import datasets

def calc_distance(point: np.array, points: np.array):
    if len(point) != len(points[0,:]):
        raise AttributeError("sizes of matrices is calc_distance do not match")
    return np.sqrt(np.sum(((points - point)**2), axis=1)) 

def target_key(point: np.array, data: np.array, target: np.array, k: int):
    distances = calc_distance(point, data[:,0:len(point)])
    key = sorted(zip(distances, target), key=lambda x: x[0])
    a, b = zip(*key)

    a = np.array(a)
    b = np.array(b)

    return b[0:k]

def count_keys(data: np.array):
    return np.unique(data, return_counts=True) 

def knn_predict(point: np.array, training_data: np.array, training_target: np.array, k: int):
    keys, counts = count_keys(target_key(point, training_data, training_target, k))
    return keys[np.argmax(counts)]

if __name__ == '__main__':

    iris = datasets.load_iris()

    prediction = knn_predict(np.array([7.5, 2.5,3.0]), iris.data, iris.target, 5)
    print(prediction)