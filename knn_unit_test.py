import unittest
from knn import KNN
from sklearn import datasets
import numpy as np

class TestKNN(unittest.TestCase):

    def setUp(self):
        iris_dataset = datasets.load_iris()
        self.Knn = KNN(iris_dataset)

    def test_calc_distans(self):
        result = self.Knn.calc_distance(np.array([1,1]), np.array([[1,0],[2,1]]))
        np.testing.assert_array_equal(result, np.array([1,1]))

    def test_calc_distans(self):
        result = self.Knn.calc_distance(np.array([1,1]), np.array([[1,0],[2,1]]))
        np.testing.assert_array_equal(result, np.array([1,1]))

if __name__ == '__main__':
    unittest.main()