import numpy as np
from sklearn import datasets
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def calc_distance(point: np.array, points: np.array) -> np.array:
    if len(point) != len(points[0,:]):
        raise AttributeError("Sizes of matrices in calc_distance do not match")
    return np.sqrt(np.sum(((points - point)**2), axis=1)) 

def target_key(point: np.array, data: np.array, target: np.array, k: int) -> np.array:
    distances = calc_distance(point, data[:,0:len(point)])
    key = sorted(zip(distances, target), key=lambda x: x[0])
    a, b = zip(*key)

    a = np.array(a)
    b = np.array(b)

    return b[0:k]

def count_keys(data: np.array) -> np.array:
    return np.unique(data, return_counts=True);

def knn_predict(point: np.array, training_data: np.array, training_target: np.array, k: int) -> np.array:
    keys, counts = count_keys(target_key(point, training_data, training_target, k))
    return keys[np.argmax(counts)]

def random_point(dimension: int, point_range=[3,7]) -> np.array:
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


    # point = random_point(3)
    # print(f"Random point: {point}")
    # prediction = knn_predict(point, iris.data, iris.target, 5)
    # print(f"Predicted: {prediction} ({iris.target_names[prediction]})")
    # plot(point, iris.data, iris.target, prediction)
    
    mnist = datasets.load_digits()
    iris = datasets.load_iris()

    k = 5

    print("---iris---")
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0, stratify=iris.target)
    predictions = np.array([knn_predict(sample, X_train, y_train, k) for sample in X_test])
    acc = np.mean(predictions == y_test)
    print(f"Accuracy after {len(predictions)} predictions: {acc} (k={k})")

    print("\n")

    print("---mnist---")
    X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=0, stratify=mnist.target)
    predictions = np.array([knn_predict(sample, X_train, y_train, k) for sample in X_test])
    acc = np.mean(predictions == y_test)
    print(f"Accuracy after {len(predictions)} predictions: {acc} (k={k})")
 