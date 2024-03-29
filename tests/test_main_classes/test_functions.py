import unittest
import numpy as np
from scipy.interpolate import interp1d

from sola.main_classes.functions import Piecewise_1D, Null_1D, Constant_1D, Random_1D, Interpolation_1D, ComplexExponential_1D, Polynomial_1D, SinusoidalPolynomial_1D, SinusoidalGaussianPolynomial_1D, NormalModes_1D, Gaussian_Bump_1D, Dgaussian_Bump_1D # noqa

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

    def test_evaluate_return_points(self):
        points, values = self.piecewise.evaluate(np.array([1, 3, 5, 7, 9, 11]),
                                                 return_points=True)
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

    def test_evaluate_return_points(self):
        points, values = self.null_1d.evaluate(np.array([1, 3, 5, 7, 9, 11]),
                                               return_points=True)
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

    def test_evaluate_return_points(self):
        points, values = self.constant_1d.evaluate(
                            np.array([1, 3, 5, 7, 9, 11]),
                            return_points=True)
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
            r, check_if_in_domain=True, return_points=True),
            expected_output)

    def test_str(self):
        self.assertEqual(str(self.interpolation), 'interpolation_1d')


class TestComplexExponential_1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 1]])
        self.frequency = 1.0
        self.func = ComplexExponential_1D(self.domain, self.frequency)

    def test_init(self):
        self.assertEqual(self.func.domain, self.domain)
        self.assertEqual(self.func.frequency, self.frequency)

    def test_evaluate(self):
        r = np.array([0.5])
        expected_output = np.exp(-2 * np.pi * self.frequency * 1j * r /
                                 self.domain.total_measure
                                 ) / self.domain.total_measure
        np.testing.assert_array_equal(self.func.evaluate(r), expected_output)

    def test_str(self):
        self.assertEqual(str(self.func), 'ComplExponential_1D')


class TestPolynomial_1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[-10, 10]])
        self.poly = Polynomial_1D(self.domain, order=2, min_val=-1, max_val=1,
                                  stretching=1, center=0)

    def test_init(self):
        self.assertEqual(self.poly.order, 2)
        self.assertEqual(self.poly.min_val, -1)
        self.assertEqual(self.poly.max_val, 1)
        self.assertEqual(self.poly.stretching, 1.0)
        self.assertEqual(self.poly.center, 0.0)

    def test_generate_random_coefficients(self):
        coefficients = self.poly.generate_random_coefficients()
        self.assertEqual(len(coefficients), self.poly.order + 1)
        self.assertTrue((coefficients >= self.poly.min_val).all())
        self.assertTrue((coefficients <= self.poly.max_val).all())

    def test_evaluate(self):
        r = np.array([1, 2, 3])
        values = self.poly.evaluate(r, check_if_in_domain=False)
        self.assertEqual(values.shape, r.shape)

    def test_evaluate_out_of_domain(self):
        r = np.array([-20, 20])
        values = self.poly.evaluate(r)
        self.assertEqual(values.size, 0)

    def test_str(self):
        self.assertEqual(str(self.poly), 'Polynomial_1D')


class TestSinusoidalPolynomial_1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 10]])
        self.sin_poly = SinusoidalPolynomial_1D(self.domain, order=2,
                                                min_val=-1, max_val=1,
                                                min_f=0, max_f=1, seed=0)

    def test_init(self):
        self.assertEqual(self.sin_poly.order, 2)
        self.assertEqual(self.sin_poly.min_val, -1)
        self.assertEqual(self.sin_poly.max_val, 1)
        self.assertEqual(self.sin_poly.min_f, 0)
        self.assertEqual(self.sin_poly.max_f, 1)
        self.assertEqual(self.sin_poly.seed, 0)

    def test_generate_random_parameters(self):
        parameters = self.sin_poly.generate_random_parameters()
        coefficients, frequencies, phases = parameters
        self.assertEqual(len(coefficients), 3)
        self.assertEqual(len(frequencies), 3)
        self.assertEqual(len(phases), 3)

    def test_poly_with_sinusoidal(self):
        x = np.array([0, 1, 2])
        result = self.sin_poly.poly_with_sinusoidal(x)
        self.assertEqual(result.shape, x.shape)

    def test_evaluate(self):
        r = np.array([0, 1, 2])
        result = self.sin_poly.evaluate(r, check_if_in_domain=False)
        self.assertEqual(result.shape, r.shape)


class TestSinusoidalGaussianPolynomial_1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 10]])
        self.order = 2
        self.min_val = -1
        self.max_val = 1
        self.min_f = 0
        self.max_f = 1
        self.spread = 1
        self.seed = 42
        self.function = SinusoidalGaussianPolynomial_1D(self.domain,
                                                        self.order,
                                                        self.min_val,
                                                        self.max_val,
                                                        self.min_f,
                                                        self.max_f,
                                                        self.spread,
                                                        self.seed)

    def test_generate_random_parameters(self):
        random_parameters = self.function.generate_random_parameters()
        coefficients, frequencies, phases = random_parameters
        self.assertEqual(len(coefficients), self.order + 1)
        self.assertEqual(len(frequencies), self.order + 1)
        self.assertEqual(len(phases), self.order + 1)
        self.assertTrue((coefficients >= self.min_val).all()
                        and (coefficients <= self.max_val).all())
        self.assertTrue((frequencies >= self.min_f).all()
                        and (frequencies <= self.max_f).all())
        self.assertTrue((phases >= 0).all() and (phases <= 2 * np.pi).all())

    def test_generate_gaussian_parameters(self):
        mean, std_dev = self.function.generate_gaussian_parameters()
        self.assertTrue(self.domain.bounds[0][0] <= mean
                        <= self.domain.bounds[0][-1])
        self.assertTrue(self.spread / 2 <= std_dev <= self.spread * 2)

    def test_evaluate(self):
        r = np.array([1, 2, 3])
        result = self.function.evaluate(r)
        self.assertEqual(len(result), len(r))

    def test_negative_order(self):
        with self.assertRaises(ValueError):
            SinusoidalGaussianPolynomial_1D(self.domain, -1, self.min_val,
                                            self.max_val, self.min_f,
                                            self.max_f, self.spread, self.seed)


class TestNormalModes_1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 1]])
        self.order = 2
        self.spread = 0.5
        self.max_freq = 1.0
        self.no_sensitivity_regions = [[0.2, 0.3]]
        self.seed = 42
        self.normal_modes = NormalModes_1D(self.domain, self.order,
                                           self.spread, self.max_freq,
                                           self.no_sensitivity_regions,
                                           self.seed)

    def test_init(self):
        self.assertEqual(self.normal_modes.order, self.order)
        self.assertEqual(self.normal_modes.seed, self.seed)
        self.assertEqual(self.normal_modes.spread, self.spread)
        self.assertEqual(self.normal_modes.max_freq, self.max_freq)
        np.testing.assert_array_equal(self.normal_modes.no_sensitivity_regions,
                                      self.no_sensitivity_regions)

    def test_generate_random_parameters(self):
        coefficients, shifts = self.normal_modes.generate_random_parameters()
        self.assertEqual(len(coefficients), self.order)
        self.assertEqual(len(shifts), self.order)

    def test_generate_function_parameters(self):
        func_parameters = self.normal_modes.generate_function_parameters()
        mean, std_dev, frequency, shift = func_parameters
        self.assertTrue(0 <= mean <= 1)
        self.assertTrue(self.spread / 2 <= std_dev <= self.spread * 2)
        self.assertTrue(0 <= frequency <= self.max_freq)
        self.assertTrue(0 <= shift <= np.pi)

    def test_evaluate(self):
        r = np.array([0.1, 0.25, 0.4])
        r_eval, result = self.normal_modes.evaluate(r, return_points=True)
        self.assertEqual(len(r_eval), len(r))
        self.assertEqual(len(result), len(r))

    def test_str(self):
        self.assertEqual(str(self.normal_modes), 'NormalModes_1D')


class TestGaussianBump1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 2]])
        self.center = 0.5
        self.width = 1.0
        self.pointiness = 2
        self.unimodularity_precision = 1000
        self.gaussian_bump = Gaussian_Bump_1D(self.domain, self.center,
                                              self.width, self.pointiness,
                                              self.unimodularity_precision)

    def test_initialization(self):
        self.assertEqual(self.gaussian_bump.center, self.center)
        self.assertEqual(self.gaussian_bump.width, self.width)
        self.assertEqual(self.gaussian_bump.pointiness, self.pointiness)
        self.assertEqual(self.gaussian_bump.unimodularity_precision,
                         self.unimodularity_precision)

    def test_compute_bump(self):
        r = np.array([0.2, 0.5, 0.8])
        bump = self.gaussian_bump._compute_bump(r)
        self.assertEqual(bump.shape, r.shape)

    def test_normalization(self):
        norm = self.gaussian_bump.normalization
        self.assertIsInstance(norm, float)

    def test_str(self):
        self.assertEqual(str(self.gaussian_bump), 'Gaussian_Bump_1D')


class TestDgaussianBump1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 10]])
        self.center = 5
        self.width = 2
        self.pointiness = 0.1
        self.unimodularity_precision = 1000
        self.dgaussian_bump_1d = Dgaussian_Bump_1D(self.domain, self.center,
                                                   self.width, self.pointiness,
                                                   self.unimodularity_precision) # noqa

    def test_init(self):
        self.assertEqual(self.dgaussian_bump_1d.center, self.center)
        self.assertEqual(self.dgaussian_bump_1d.width, self.width)
        self.assertEqual(self.dgaussian_bump_1d.pointiness, self.pointiness)
        self.assertEqual(self.dgaussian_bump_1d.unimodularity_precision,
                         self.unimodularity_precision)
        self.assertIsInstance(self.dgaussian_bump_1d.bump, Gaussian_Bump_1D)

    def test_compute_multiplier(self):
        r_compact_centered = np.array([1, 2, 3])
        expected_multiplier = np.array([0.0, -0.844444444, -0.69375])
        actual_multiplier = self.dgaussian_bump_1d._compute_multiplier(r_compact_centered) # noqa
        np.testing.assert_almost_equal(actual_multiplier, expected_multiplier)

    def test_evaluate(self):
        r = np.array([4, 5, 6])
        expected_dbump = np.array([0.0, 0.0, 0.0])
        actual_dbump = self.dgaussian_bump_1d.evaluate(r, check_if_in_domain=False) # noqa
        np.testing.assert_almost_equal(actual_dbump, expected_dbump)


if __name__ == '__main__':
    unittest.main()
