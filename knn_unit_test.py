import unittest
from knn import KNN
from sklearn import datasets
import numpy as np

class TestKNN(unittest.TestCase):

    def setUp(self):
        self.data_set=np.array([[0,6,0,6,3,7],[7,7,3,3,1,0]]).T
        self.target_set = np.array([0, 0, 1, 1, 0, 1])
        self.point = np.array([3,3])
        self.k = 3
        self.Knn = KNN(self.data_set, self.target_set)

    def test_calc_distans(self):
        result = self.Knn.calc_distance(self.point, self.data_set)
        np.testing.assert_array_equal(result, np.array([5,5,3,3,2,5]))

    def test_target_key(self):
        result = self.Knn.target_key(self.point, self.data_set,self.target_set, self.k)
        np.testing.assert_array_equal(result, np.array([0,1,1]))

    def test_count_keys(self):
        result = self.Knn.count_keys(self.target_set)
        np.testing.assert_array_equal(result,np.array([[0,1],[3,3]]))

    def test_predict_single(self):
        result = self.Knn.predict_single(self.point, self.data_set, self.target_set, self.k)
        np.testing.assert_array_equal(result,np.array([1]))

    def test_random_point(self):
        leng = len(self.point)
        result = self.Knn.random_point(leng)
        self.assertEqual(len(result),leng)
        for i in range(leng):
            self.assertLessEqual(result[i], 7)
            self.assertGreaterEqual(result[i], 3)

        
    # def predict(self, k:int) -> None:
# 
    # def print_accuracy(self) -> None:
# 
    # def plot_accuracy(self, ax, title) ->None:
# 
    # def plot(self, point: np.array, training_data: np.array, training_target: np.array, prediction: int)-> None:
# 
if __name__ == '__main__':
    unittest.main()