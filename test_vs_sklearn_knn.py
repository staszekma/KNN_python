import numpy as np
from knn import KNN
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from matplotlib import pyplot as plt

if __name__ == '__main__':
    mnist_dataset = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff') 

    data_size_limit = 10000
    max_k = 50

    mnist_data = mnist_dataset.data[:data_size_limit].astype(np.float32)
    mnist_target = mnist_dataset.target[:data_size_limit].astype(int)
    sk_x_train, sk_x_test, sk_y_train, sk_y_test = train_test_split(mnist_data, mnist_target, test_size=0.2, random_state=0, stratify=mnist_target)
    
    # first, the sklearn knn goes
    print("----- SKLEARN KNN -----")
    sk_accuracy = []
    sk_single_time = []
    sk_timer_start = timer()
    for k in range(1, max_k + 1):
        sk_single_time_start = timer()
        sklearn_knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', metric='euclidean', weights='uniform')
        sklearn_knn.fit(sk_x_train, sk_y_train)
        predictions = sklearn_knn.predict(sk_x_test)
        sk_single_time_stop = timer()
        sk_single_pred_time = sk_single_time_stop - sk_single_time_start
        sk_single_time.append(sk_single_pred_time)
        accuracy = np.mean(predictions == sk_y_test)
        sk_accuracy.append(accuracy)
        print(f"Accuracy: {accuracy} (k={k})")
    sk_timer_stop = timer()
    sk_time = sk_timer_stop - sk_timer_start
    print(f"Predicting with k from 1 to {max_k} took sklearn knn {sk_time}s")

    # now, our knn goes
    print("----- OUR KNN -----")
    our_timer_start = timer()
    my_knn = KNN(mnist_data, mnist_target)
    our_single_time = []
    our_accuracy = []
    for k in range(1, max_k + 1):
        our_single_time_start = timer()
        my_knn.predict(k) 
        our_single_time_stop = timer()
        our_single_pred_time = our_single_time_stop - our_single_time_start
        our_single_time.append(our_single_pred_time)
        accuracy = np.mean(my_knn.predictions == my_knn.y_test)
        our_accuracy.append(accuracy)
        print(f"Accuracy: {accuracy} (k={k})")
    our_timer_stop = timer()
    our_time = our_timer_stop - our_timer_start
    print(f"Predicting with k from 1 to {max_k} took our knn {our_time}s")
 
    fig, (ax_our, ax_sk, ax_diff) = plt.subplots(1, 3, figsize=(12, 5))

    ax_our.plot(range(1, max_k + 1), our_accuracy)
    ax_our.set_title("Our KNN accuracy vs K")
    ax_our.set_xlabel("k")
    ax_our.set_ylabel("accuracy")

    ax_sk.plot(range(1, max_k + 1), sk_accuracy)
    ax_sk.set_title("sklearn KNN accuracy vs K")
    ax_sk.set_xlabel("k")
    ax_sk.set_ylabel("accuracy")

    ax_diff.plot(range(1, max_k + 1), np.array(sk_accuracy) - np.array(our_accuracy))
    ax_diff.set_title("Accuracy difference\n(sk - our)")
    ax_diff.set_xlabel("k")
    ax_diff.set_ylabel("accuracy diff")

 
    fig.tight_layout()
    plt.title(f"Our KNN (finished in {our_time}s)\nvs\n sklearn KNN (finished in {sk_time}s)")
    plt.show()
    
