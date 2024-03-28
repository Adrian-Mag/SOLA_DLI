import unittest
import numpy as np
from scipy.interpolate import interp1d

from sola.main_classes.functions import Piecewise_1D, Null_1D, Constant_1D
from sola.main_classes.functions import Random_1D, Interpolation_1D

from sola.main_classes.domains import HyperParalelipiped


class TestPiecewise_1D(unittest.TestCase):
    def setUp(self):

        self.domain = HyperParalelipiped([[0, 10]])
        self.intervals = np.array([0, 2, 4, 6, 8, 10])
        self.values = np.array([1, 2, 3, 4, 5])
        self.piecewise = Piecewise_1D(self.domain, self.intervals, self.values)

    def test_evaluate_single_point(self):
        self.assertEqual(self.piecewise.evaluate(1), np.array([1]))

    def test_evaluate_multiple_points(self):
        r = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(self.piecewise.evaluate(r),
                                      np.array([1, 2, 3, 4, 5]))

    def test_evaluate_out_of_domain(self):
        self.assertEqual(self.piecewise.evaluate(11, check_if_in_domain=False),
                         np.array([5]))
        np.testing.assert_array_equal(self.piecewise.evaluate(
                                        11, check_if_in_domain=True
                                        ), np.array([]))

    def test_evaluate_return_in_domain(self):
        points, values = self.piecewise.evaluate(np.array([1, 3, 5, 7, 9, 11]),
                                                 return_in_domain=True)
        np.testing.assert_array_equal(points, np.array([1, 3, 5, 7, 9]))
        np.testing.assert_array_equal(values, np.array([1, 2, 3, 4, 5]))

    def test_evaluate_no_check_domain(self):
        np.testing.assert_array_equal(
            self.piecewise.evaluate(np.array([1, 3, 5, 7, 9, 11]),
                                    check_if_in_domain=False),
            np.array([1, 2, 3, 4, 5, 5])
            )


class TestNull_1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 10]])
        self.null_1d = Null_1D(self.domain)

    def test_evaluate_single_point(self):
        self.assertEqual(self.null_1d.evaluate(1), np.array([0]))

    def test_evaluate_multiple_points(self):
        r = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(self.null_1d.evaluate(r),
                                      np.zeros(5))

    def test_evaluate_out_of_domain(self):
        self.assertEqual(self.null_1d.evaluate(11, check_if_in_domain=False),
                         np.array([0]))
        np.testing.assert_array_equal(self.null_1d.evaluate(
                                      11, check_if_in_domain=True),
                                      np.array([]))

    def test_evaluate_return_in_domain(self):
        points, values = self.null_1d.evaluate(np.array([1, 3, 5, 7, 9, 11]),
                                               return_in_domain=True)
        np.testing.assert_array_equal(points, np.array([1, 3, 5, 7, 9]))
        np.testing.assert_array_equal(values, np.zeros(5))

    def test_evaluate_no_check_domain(self):
        np.testing.assert_array_equal(
            self.null_1d.evaluate(np.array([1, 3, 5, 7, 9, 11]),
                                  check_if_in_domain=False), np.zeros(6))


class TestConstant_1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 10]])
        self.constant_1d = Constant_1D(self.domain, value=3)

    def test_evaluate_single_point(self):
        self.assertEqual(self.constant_1d.evaluate(1), np.array([3]))

    def test_evaluate_multiple_points(self):
        r = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_equal(self.constant_1d.evaluate(r),
                                      np.full(5, 3))

    def test_evaluate_out_of_domain(self):
        self.assertEqual(self.constant_1d.evaluate(11,
                                                   check_if_in_domain=False),
                         np.array([3]))
        np.testing.assert_array_equal(self.constant_1d.evaluate(
                                      11, check_if_in_domain=True),
                                      np.array([]))

    def test_evaluate_return_in_domain(self):
        points, values = self.constant_1d.evaluate(
                            np.array([1, 3, 5, 7, 9, 11]),
                            return_in_domain=True)
        np.testing.assert_array_equal(points, np.array([1, 3, 5, 7, 9]))
        np.testing.assert_array_equal(values, np.full(5, 3))

    def test_evaluate_no_check_domain(self):
        np.testing.assert_array_equal(
            self.constant_1d.evaluate(
                            np.array([1, 3, 5, 7, 9, 11]),
                            check_if_in_domain=False), np.full(6, 3))


class TestRandom1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped(bounds=[[0, 1]])
        self.random_1d = Random_1D(self.domain, seed=42)

    def test_init(self):
        self.assertEqual(self.random_1d.seed, 42)
        self.assertEqual(self.random_1d.continuous, False)
        self.assertIsNone(self.random_1d.boundaries)
        self.assertIsNotNone(self.random_1d.function)

    def test_create_function(self):
        self.assertIsNotNone(self.random_1d._create_function())

    def test_determine_segments(self):
        self.assertIsInstance(self.random_1d._determine_segments(), int)

    def test_determine_inpoints(self):
        segments = self.random_1d._determine_segments()
        inpoints = self.random_1d._determine_inpoints(segments)
        self.assertIsInstance(inpoints, list)
        self.assertEqual(len(inpoints), segments - 1)

    def test_create_partitions(self):
        inpoints = [0.2, 0.4, 0.6, 0.8]
        partitions = self.random_1d._create_partitions(inpoints)
        self.assertIsInstance(partitions, list)
        self.assertEqual(len(partitions), len(inpoints) + 1)

    def test_create_model(self):
        partition = (0, 1)
        model = self.random_1d._create_model(partition)
        self.assertIsInstance(model, np.ndarray)

    def test_evaluate(self):
        r = 0.5
        value = self.random_1d.evaluate(r)
        self.assertIsInstance(value, np.ndarray)


class TestInterpolation_1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 10]])
        self.values = np.array([1, 2, 3])
        self.raw_domain = np.array([1, 2, 3])
        self.interpolation = Interpolation_1D(self.values, self.raw_domain,
                                              self.domain)

    def test_evaluate(self):
        r = np.array([1, 2])
        expected_output = interp1d(self.raw_domain, self.values, kind='linear',
                                   fill_value='extrapolate')(r)
        np.testing.assert_array_equal(self.interpolation.evaluate(
                    r, check_if_in_domain=False), expected_output)

    def test_evaluate_with_in_domain(self):
        r = np.array([1, 2])
        expected_output = r, interp1d(self.raw_domain, self.values,
                                      kind='linear',
                                      fill_value='extrapolate')(r)
        np.testing.assert_array_equal(self.interpolation.evaluate(
            r, check_if_in_domain=True, return_in_domain=True),
            expected_output)

    def test_str(self):
        self.assertEqual(str(self.interpolation), 'interpolation_1d')


if __name__ == '__main__':
    unittest.main()
