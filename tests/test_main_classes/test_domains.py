import unittest
import numpy as np
from sola.main_classes.domains import HyperParalelipiped


class TestHyperParalelipiped(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 1]])
        self.domain2 = HyperParalelipiped([[0, 10], [0, 10]])

    def test_dynamic_mesh(self):
        self.assertTrue(np.array_equal(self.domain.dynamic_mesh(10),
                                       np.linspace(0, 1, 10)))

    def test_dynamic_mesh_single_dimension(self):
        self.assertTrue(np.array_equal(self.domain.dynamic_mesh(5),
                                       np.linspace(0, 1, 5)))

    def test_dynamic_mesh_multiple_dimensions(self):
        expected_result = np.meshgrid(np.linspace(0, 10, 5),
                                      np.linspace(0, 10, 5))
        actual_result = self.domain2.dynamic_mesh(5)
        for expected, actual in zip(expected_result, actual_result):
            self.assertTrue(np.array_equal(expected, actual))

    def test_dynamic_mesh_zero_samples(self):
        with self.assertRaises(ValueError):
            self.domain.dynamic_mesh(0)

    def test_dynamic_mesh_negative_samples(self):
        with self.assertRaises(ValueError):
            self.domain.dynamic_mesh(-5)

    def test_sample_domain(self):
        samples = self.domain.sample_domain(10)
        self.assertEqual(samples.shape, (10, 1))
        self.assertTrue(np.all((0 <= samples) & (samples <= 1)))

        sample = self.domain.sample_domain()
        self.assertEqual(sample.shape, (1, 1))
        self.assertTrue(0 <= sample <= 1)

        samples = self.domain2.sample_domain(10)
        self.assertEqual(samples.shape, (10, 2))
        self.assertTrue(np.all((0 <= samples) & (samples <= 10)))

        sample = self.domain2.sample_domain()
        self.assertEqual(sample.shape, (1, 2))
        self.assertTrue(np.all((0 <= sample) & (sample <= 10)))

    def test_check_if_in_domain(self):
        self.assertTrue(self.domain.check_if_in_domain(0.5))
        self.assertFalse(self.domain.check_if_in_domain(1.5))

    def test_check_if_in_domain_single_value(self):
        self.assertTrue(self.domain2.check_if_in_domain(np.array([5, 5])))
        self.assertFalse(self.domain2.check_if_in_domain(np.array([15, 5])))
        self.assertFalse(self.domain2.check_if_in_domain(np.array([5, 15])))

    def test_check_if_in_domain_array(self):
        test_values = np.array([[5, 5], [15, 5], [5, 15]])
        expected_result = [True, False, False]
        self.assertEqual(self.domain2.check_if_in_domain(test_values),
                         expected_result)

    def test_check_if_in_domain_wrong_dimension(self):
        with self.assertRaises(Exception):
            self.domain2.check_if_in_domain(np.array([5, 5, 5]))

    def test_check_if_in_domain_wrong_type(self):
        with self.assertRaises(Exception):
            self.domain2.check_if_in_domain('5, 5')

    def test_check_if_in_domain_1d_array(self):
        # Test with a numpy array that is within the domain
        values = np.array([0.5, 0.2, 0.8])
        expected_result = np.array([True, True, True])
        np.testing.assert_array_equal(self.domain.check_if_in_domain(values),
                                      expected_result)

        # Test with a numpy array that is not within the domain
        values = np.array([1.5, 2.0, 0.8])
        expected_result = np.array([False, False, True])
        np.testing.assert_array_equal(self.domain.check_if_in_domain(values),
                                      expected_result)

        # Test with a numpy array that has a wrong dimension
        values = np.array([[0.5, 0.2], [0.8, 1.0]])
        with self.assertRaises(Exception):
            self.domain.check_if_in_domain(values)

    def test_check_if_in_domain_wrong_inner_dimension(self):
        # Test with a numpy array where one of the inner arrays has a wrong
        # dimension
        values = np.array([[5, 5, 5], [15, 5, 5], [5, 15, 5]])
        with self.assertRaises(Exception):
            self.domain2.check_if_in_domain(values)

    def test_eq(self):
        # Create a new instance with the same bounds as self.domain
        same_domain = HyperParalelipiped([[0, 1]])
        self.assertTrue(self.domain == same_domain)

        # Create a new instance with different bounds
        different_domain = HyperParalelipiped([[0, 2]])
        self.assertFalse(self.domain == different_domain)

        # Create a new instance with the same bounds but in a different order
        different_order_domain = HyperParalelipiped([[1, 0]])
        self.assertFalse(self.domain == different_order_domain)

        # Test with a non-HyperParalelipiped instance
        not_a_domain = "not a domain"
        self.assertFalse(self.domain == not_a_domain)

        # Create a new instance with the same bounds as self.domain2
        same_domain = HyperParalelipiped([[0, 10], [0, 10]])
        self.assertTrue(self.domain2 == same_domain)

        # Create a new instance with different bounds
        different_domain = HyperParalelipiped([[0, 10], [0, 20]])
        self.assertFalse(self.domain2 == different_domain)

        # Create a new instance with the same bounds but in a different order
        different_order_domain = HyperParalelipiped([[0, 10], [10, 0]])
        self.assertFalse(self.domain2 == different_order_domain)

        # Test with a non-HyperParalelipiped instance
        not_a_domain = "not a domain"
        self.assertFalse(self.domain2 == not_a_domain)

    def test_total_measure(self):
        # Test with a 1D domain
        domain = HyperParalelipiped(bounds=[[0, 1]])
        self.assertEqual(domain.total_measure, 1)

        # Test with a 2D domain
        domain = HyperParalelipiped(bounds=[[0, 1], [0, 2]])
        self.assertEqual(domain.total_measure, 2)

        # Test with a 3D domain
        domain = HyperParalelipiped(bounds=[[0, 1], [0, 2], [0, 3]])
        self.assertEqual(domain.total_measure, 6)


if __name__ == '__main__':
    unittest.main()
