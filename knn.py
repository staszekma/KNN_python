import numpy as np
from sklearn import datasets
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

class KNN:
    def __init__(self, dataset):
        self.dataset = dataset
        self.last_acc = [] 
        self.last_k = []
        self.predictions = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset.data, self.dataset.target, test_size=0.2, random_state=0, stratify=self.dataset.target)
 
    def calc_distance(self, point: np.array, points: np.array) -> np.array:
        if len(point) != len(points[0,:]):
            raise AttributeError("Sizes of matrices in calc_distance do not match")
        return np.sqrt(np.sum(((points - point)**2), axis=1)) 

    def target_key(self, point: np.array, data: np.array, target: np.array, k:int) -> np.array:
        distances = self.calc_distance(point, data[:,0:len(point)])
        # key = sorted(zip(distances, target), key=lambda x: x[0])
        # a, b = zip(*key)
        #
        # a = np.array(a)
        # b = np.array(b)
        #
        # return b[0:k]
        target_indices = np.argpartition(distances, k-1)[:k] # this returs indices of the k smallest distances, unsorted
        return target[target_indices]

    def count_keys(self, data: np.array) -> np.array:
        return np.unique(data, return_counts=True);

    def predict_single(self, point: np.array, training_data: np.array, training_target: np.array, k: int) -> np.array:
        keys, counts = self.count_keys(self.target_key(point, training_data, training_target, k))
        return keys[np.argmax(counts)]

    def random_point(self, dimension: int, point_range=[3,7]) -> np.array:
        return np.array([random.uniform(*point_range) for _ in range (dimension)])

    def predict(self, k:int) -> None:
        data_size = len(self.X_train)
        if(k > len(self.X_train)):
            print(f"K of {k} is larger than dataset size of {data_size}, clamped to data_size")
            k = data_size - 1
        self.last_k.append(k)
        self.predictions = np.array([self.predict_single(sample, self.X_train, self.y_train, k) for sample in self.X_test])
        self.last_acc.append(np.mean(self.predictions == self.y_test))

    def print_accuracy(self) -> None:
        if(self.predictions is None):
            print("Run predict() first")
            return
        print(f"Accuracy after {len(self.predictions)} predictions: {self.last_acc[-1]} (k={self.last_k[-1]})")

    def plot_accuracy(self, ax, title) ->None:
        ax.plot(self.last_k, self.last_acc)
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.set_ylabel("accuracy")

        
    def plot(self, point: np.array, training_data: np.array, training_target: np.array, prediction: int)-> None:
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
    
    mnist_dataset = datasets.load_digits()
    iris_dataset = datasets.load_iris()

    iris = KNN(iris_dataset)
    mnist = KNN(mnist_dataset)

    max_k = 100

    print("---iris---")
    for k in range(1,max_k):
        iris.predict(k)
        iris.print_accuracy()

    print("\n")

    print("---mnist---")
    for k in range(1,max_k):
        mnist.predict(k)
        mnist.print_accuracy()

    fig, (ax_iris, ax_mnist) = plt.subplots(1, 2, figsize=(12, 5))
    iris.plot_accuracy(ax_iris, "Iris")
    mnist.plot_accuracy(ax_mnist, "Mnist")
    fig.tight_layout()
    plt.show()
