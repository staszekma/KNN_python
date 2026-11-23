import numpy as np
from sklearn import datasets
import random
from matplotlib import pyplot as plt

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

def random_point(dimension: int, point_range=[3,7]):
    return np.array([random.uniform(*point_range) for _ in range (dimension)])

def plot(point: np.array, training_data: np.array, training_target: np.array, prediction: int):
    dim = len(point)
    pro = None
    if dim == 1:
        training_data[:,1]=np.zeros_like(training_data[:,0])
        point = np.append(point, 0)
        dim = 2
    elif dim == 2:
        pro = None
    elif dim == 3:
        pro = '3d'
    else:
        raise AttributeError("bad point dimension to plot")
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = pro)
    ax.scatter(*training_data[:,0:dim].T, c=training_target)
    ax.scatter(*point, c="green", s=160)
    ax.scatter(*point, vmin=min(training_target), vmax=max(training_target), c=prediction)
    plt.show()

if __name__ == '__main__':

    iris = datasets.load_iris()

    point = random_point(3)
    print(point)
    prediction = knn_predict(point, iris.data, iris.target, 5)
    print(prediction)

    plot(point, iris.data, iris.target, prediction)