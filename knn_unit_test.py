import unittest
from knn import KNN
from sklearn import datasets
import numpy as np
import io
import contextlib
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt

class TestKNN(unittest.TestCase):

    def setUp(self):
        self.data_set=np.array([[0,6,0,6,3,7],[7,7,3,3,1,0]]).T
        self.target_set = np.array([0, 0, 1, 1, 0, 1])
        self.point = np.array([3,3])
        self.k = 3
        self.Knn = KNN(self.data_set, self.target_set)


    def test_random_point(self):
        leng = len(self.point)
        result = self.Knn.random_point(leng)
        self.assertEqual(len(result),leng)
        for i in range(leng):
            self.assertLessEqual(result[i], 7)
            self.assertGreaterEqual(result[i], 3)
    
    def test_calc_distans(self):
        result = self.Knn.calc_distance(self.point, self.data_set)
        np.testing.assert_array_equal(result, np.array([25,25,9,9,4,25]))
    
    def test_target_key(self):
        result = self.Knn.target_key(self.point, self.data_set,self.target_set)
        np.testing.assert_array_equal(result, np.array([0,1,1,0,0,1]))

    def test_find_neighbours(self):
        self.Knn.find_neighbours()
        np.testing.assert_array_equal(self.Knn.neighbours, np.array([[1,0,0,1],[1,0,0,1]]))

    def test_count_keys(self):
        result = self.Knn.count_keys(self.target_set)
        np.testing.assert_array_equal(result,np.array([[0,1],[3,3]]))

    def test_vote(self):
        result = self.Knn.vote(self.target_set, self.k)
        np.testing.assert_array_equal(result,np.array([0]))
    
    def test_predict(self):
        self.Knn.predict(self.k)
        np.testing.assert_array_equal(self.Knn.predictions,np.array([0, 0]))

    def test_print_accuracy_before_prediction(self):
        with io.StringIO() as output:
            with contextlib.redirect_stdout(output):
                self.Knn.print_accuracy()
            self.assertEqual(
                output.getvalue(),
                "Run predict() first\n"
            )

    def test_print_accuracy_after_prediction(self):
        self.Knn.predict(3)

        with io.StringIO() as output:
            with contextlib.redirect_stdout(output):
                self.Knn.print_accuracy()

            text = output.getvalue()
            self.assertIn("predictions", text)
            self.assertIn("k=3", text)

    def test_plot_accuracy(self):
        ax = Mock()
        self.Knn.last_k = [1, 3, 5]
        self.Knn.last_acc = [0.5, 0.75, 1.0]

        self.Knn.plot_accuracy(ax, "Test accuracy")

        ax.plot.assert_called_once_with(
            [1, 3, 5],
            [0.5, 0.75, 1.0]
        )
        ax.set_title.assert_called_once_with("Test accuracy")
        ax.set_xlabel.assert_called_once_with("k")
        ax.set_ylabel.assert_called_once_with("accuracy")


if __name__ == '__main__':
    unittest.main()