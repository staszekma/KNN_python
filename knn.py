import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_openml
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer

def func_time(func): 
    def wrapper(*args, **kwargs):
        t1 = timer();
        func(*args, **kwargs);
        t2 = timer();
        print(f"the {func.name} took {t2-t1}")
    return wrapper


class KNN:
    def __init__(self, dataset, target):
        self.data= np.asarray(dataset)
        self.target = np.asarray(target)
        self.last_acc = [] 
        self.last_k = []
        self.predictions = None
        self.neighbours = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=0, stratify=self.target)
 
    def calc_distance(self, point: np.ndarray, points: np.ndarray) -> np.array:
        if len(point) != len(points[0,:]):
            raise AttributeError("Sizes of matrices in calc_distance do not match")
        return np.sqrt(np.sum(((points - point)**2), axis=1)) 

    def target_key(self, point: np.ndarray, data: np.ndarray, target: np.ndarray) -> np.ndarray:
        distances = self.calc_distance(point, data[:,0:len(point)])
        # key = sorted(zip(distances, target), key=lambda x: x[0])
        # a, b = zip(*key)
        #
        # a = np.array(a)
        # b = np.array(b)
        #
        # return b[0:k]
        # target_indices = np.argpartition(distances, k-1)[:k] # this returs indices of the k smallest distances, unsorted
        target_indices = np.argsort(distances)
        return target[target_indices]

    def find_neighbours(self) -> None:
        # neighbours is a 2d array, that stores all the neighbours for a X_train sample that coresponds to the row
        self.neighbours = np.array([self.target_key(sample, self.X_train, self.y_train) for sample in self.X_test])        

    def count_keys(self, data: np.ndarray) -> np.ndarray:
        return np.unique(data, return_counts=True)

    def vote(self, neighbours:np.ndarray, k: int) -> np.ndarray:
        keys, counts = self.count_keys(neighbours[:k])
        return keys[np.argmax(counts)]

    def random_point(self, dimension: int, point_range=[3,7]) -> np.ndarray:
        return np.array([random.uniform(*point_range) for _ in range (dimension)])

    # @func_time
    def predict(self, k:int) -> None:
        if self.neighbours is None:
            self.find_neighbours()
        data_size = self.neighbours.shape[1]
        if(k > data_size):
            print(f"K of {k} is larger than dataset size of {data_size}, clamped to data_size")
            k = data_size
        self.last_k.append(k)
        self.predictions = np.array([self.vote(row, k) for row in self.neighbours])
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
    
    # mnist_dataset = datasets.load_digits()
    
    # the parser is set, because pandas parser used by openml is moody
    mnist_dataset = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff') 
    iris_dataset = datasets.load_iris()

    data_size_limit = 10000

    mnist_data = mnist_dataset.data[:data_size_limit].astype(np.float32)
    mnist_target = mnist_dataset.target[:data_size_limit].astype(int)


    iris = KNN(iris_dataset.data, iris_dataset.target)
    mnist = KNN(mnist_data, mnist_target)

    max_k = 50

    print("---iris---")
    for k in range(1,max_k+1):
        iris.predict(k)
        iris.print_accuracy()

    print("\n")

    print("---mnist---")
    for k in range(1,max_k+1):
        mnist.predict(k)
        mnist.print_accuracy()

    fig, (ax_iris, ax_mnist) = plt.subplots(1, 2, figsize=(12, 5))
    iris.plot_accuracy(ax_iris, "Iris")
    mnist.plot_accuracy(ax_mnist, "Mnist")
    fig.tight_layout()
    plt.show()
