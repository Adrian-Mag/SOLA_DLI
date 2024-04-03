import unittest
import numpy as np
from scipy.interpolate import interp1d

from sola.main_classes.functions import Piecewise_1D, Null_1D, Constant_1D, Random_1D, Interpolation_1D, ComplexExponential_1D, Polynomial_1D, SinusoidalPolynomial_1D, SinusoidalGaussianPolynomial_1D, NormalModes_1D, Gaussian_Bump_1D, Dgaussian_Bump_1D, Gaussian_1D, Moorlet_1D, Boxcar_1D, Haar_1D, Ricker_1D, Dgaussian_1D, Bump_1D, Dbump_1D, Triangular_1D, Fourier # noqa

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
        r_compact_centered = np.array([0, 2, 3])
        expected_multiplier = np.array([0.0, -0.844444444, -0.69375])
        actual_multiplier = self.dgaussian_bump_1d._compute_multiplier(r_compact_centered) # noqa
        np.testing.assert_almost_equal(actual_multiplier, expected_multiplier)

    def test_evaluate(self):
        r = np.array([4, 5, 6])
        expected_dbump = np.array([0.0, 0.0, 0.0])
        actual_dbump = self.dgaussian_bump_1d.evaluate(r, check_if_in_domain=False) # noqa
        np.testing.assert_almost_equal(actual_dbump, expected_dbump)


class TestGaussian1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 10]])
        self.center = 5
        self.width = 2
        self.gaussian = Gaussian_1D(self.domain, self.center, self.width)

    def test_initialization(self):
        self.assertEqual(self.gaussian.center, self.center)
        self.assertEqual(self.gaussian.width, self.width)
        self.assertEqual(self.gaussian.spread, self.width /
                         (5 * np.sqrt(2 * np.log(2))))

    def test_evaluate(self):
        points, values = self.gaussian.evaluate(np.array([1, 2, 3]),
                                                return_points=True)
        self.assertTrue(np.all(points == np.array([1, 2, 3])))
        self.assertTrue(np.all(values == (1 / (self.gaussian.spread * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((points - self.center) / self.gaussian.spread) ** 2))) # noqa

    def test_str(self):
        self.assertEqual(str(self.gaussian), 'Gaussian_1D')


class TestMoorlet1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 10]])
        self.center = 5
        self.spread = 2
        self.frequency = 1
        self.moorlet = Moorlet_1D(self.domain, self.center,
                                  self.spread, self.frequency)

    def test_initialization(self):
        self.assertEqual(self.moorlet.center, self.center)
        self.assertEqual(self.moorlet.spread, self.spread)
        self.assertEqual(self.moorlet.frequency, self.frequency)

    def test_compute_normalization(self):
        moorlet_vector = np.cos(self.frequency * (self.domain.dynamic_mesh(self.moorlet.unimodularity_precision) - self.center)) \
            * np.exp(-0.5 * ((self.domain.dynamic_mesh(self.moorlet.unimodularity_precision) - self.center) / self.spread) ** 2) # noqa
        area = np.trapz(moorlet_vector, self.domain.dynamic_mesh(self.moorlet.unimodularity_precision)) # noqa
        self.assertEqual(self.moorlet.normalization, area)

    def test_evaluate(self):
        points, values = self.moorlet.evaluate(np.array([1, 2, 3]),
                                               return_points=True)
        self.assertTrue(np.all(points == np.array([1, 2, 3])))
        moorlet_vector = np.cos(self.frequency * (points - self.center)) \
            * np.exp(-0.5 * ((points - self.center) / self.spread) ** 2)
        self.assertTrue(np.all(values == moorlet_vector /
                               self.moorlet.normalization))

    def test_str(self):
        self.assertEqual(str(self.moorlet), 'Moorlet_1D')


class TestHaar1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 2]])
        self.center = 1.0
        self.width = 1.0
        self.haar = Haar_1D(self.domain, self.center, self.width)

    def test_init(self):
        self.assertEqual(self.haar.center, self.center)
        self.assertEqual(self.haar.width, self.width)

    def test_evaluate(self):
        r = np.array([0.0, 0.75, 1.25, 1.75])
        expected_values = np.array([0.0, -4.0, 4.0, 0.0])
        np.testing.assert_array_equal(self.haar.evaluate(r), expected_values)

    def test_evaluate_with_return_points(self):
        r = np.array([0.0, 0.75, 1.25, 1.75])
        expected_points = np.array([0.0, 0.75, 1.25, 1.75])
        expected_values = np.array([0.0, -4.0, 4.0, 0.0])
        points, values = self.haar.evaluate(r, check_if_in_domain=False,
                                            return_points=True)
        np.testing.assert_array_equal(points, expected_points)
        np.testing.assert_array_equal(values, expected_values)

    def test_str(self):
        self.assertEqual(str(self.haar), 'Haar_1D')


class TestRicker1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 10]])
        self.center = 5
        self.width = 2
        self.ricker = Ricker_1D(self.domain, self.center, self.width)

    def test_init(self):
        self.assertEqual(self.ricker.center, self.center)
        self.assertEqual(self.ricker.width, self.width)

    def test_evaluate(self):
        r = np.array([1, 2, 3, 4, 5])
        vector = self.ricker.evaluate(r)
        self.assertEqual(vector.shape, r.shape)

    def test_evaluate_outside_domain(self):
        r = np.array([-1, 11])
        vector = self.ricker.evaluate(r, check_if_in_domain=False)
        self.assertEqual(vector.shape, r.shape)

    def test_str(self):
        self.assertEqual(str(self.ricker), 'Ricker_1D')


class TestDgaussian_1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 10]])
        self.center = 5
        self.width = 2
        self.dgaussian = Dgaussian_1D(self.domain, self.center, self.width)

    def test_init(self):
        self.assertEqual(self.dgaussian.center, self.center)
        self.assertEqual(self.dgaussian.width, self.width)
        self.assertAlmostEqual(self.dgaussian.spread, self.width /
                               (5 * np.sqrt(2 * np.log(2))), places=7)

    def test_evaluate(self):
        r = np.array([4, 5, 6])
        result = self.dgaussian.evaluate(r)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, r.shape)

    def test_evaluate_return_points(self):
        r = np.array([4, 5, 6])
        points, result = self.dgaussian.evaluate(r, return_points=True)
        self.assertIsInstance(points, np.ndarray)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(points.shape, r.shape)
        self.assertEqual(result.shape, r.shape)

    def test_str(self):
        self.assertEqual(str(self.dgaussian), 'Dgaussian_1D')


class TestBoxcar1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[-1, 1]])
        self.center = 0
        self.width = 1
        self.boxcar = Boxcar_1D(self.domain, self.center, self.width)

    def test_init(self):
        self.assertEqual(self.boxcar.center, self.center)
        self.assertEqual(self.boxcar.width, self.width)
        self.assertEqual(self.boxcar.unimodularity_precision, 1000)

    def test_evaluate(self):
        points, values = self.boxcar.evaluate(np.array([-0.25, 0, 0.25]),
                                              return_points=True)
        np.testing.assert_array_equal(points, np.array([-0.25, 0, 0.25]))
        np.testing.assert_array_equal(values, np.array([1, 1, 1]))

        values = self.boxcar.evaluate(np.array([-0.25, 0, 0.25]))
        np.testing.assert_array_equal(values, np.array([1, 1, 1]))

    def test_evaluate_outside_domain(self):
        points, values = self.boxcar.evaluate(np.array([-2, 2]),
                                              return_points=True)
        np.testing.assert_array_equal(points, np.array([]))
        np.testing.assert_array_equal(values, np.array([]))

        values = self.boxcar.evaluate(np.array([-2, 2]), return_points=False)
        np.testing.assert_array_equal(values, np.array([]))

    def test_evaluate_without_checking_domain(self):
        points, values = self.boxcar.evaluate(np.array([-2, 2]),
                                              check_if_in_domain=False,
                                              return_points=True)
        np.testing.assert_array_equal(points, np.array([-2, 2]))
        np.testing.assert_array_equal(values, np.array([0, 0]))

        values = self.boxcar.evaluate(np.array([-2, 2]),
                                      check_if_in_domain=False,
                                      return_points=False)
        np.testing.assert_array_equal(values, np.array([0, 0]))

    def test_str(self):
        self.assertEqual(str(self.boxcar), 'Boxcar_1D')


class TestBump1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[-1, 1]])
        self.center = 0.0
        self.width = 1.0
        self.bump = Bump_1D(self.domain, self.center, self.width)

    def test_init(self):
        self.assertEqual(self.bump.center, self.center)
        self.assertEqual(self.bump.width, self.width)
        self.assertIsNotNone(self.bump.normalization)

    def test_compute_normalization(self):
        normalization = self.bump._compute_normalization()
        self.assertIsNotNone(normalization)
        self.assertEqual(self.bump.normalization, normalization)

    def test_evaluate(self):
        r = np.array([0.0, 0.5, 0.9])
        values = self.bump.evaluate(r)
        self.assertEqual(values.shape, r.shape)

    def test_str(self):
        self.assertEqual(str(self.bump), 'Bump_1D')


class TestDbump_1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[-1, 1]])
        self.center = 0.0
        self.width = 1.0
        self.unimodularity_precision = 1000
        self.dbump = Dbump_1D(self.domain, self.center, self.width,
                              self.unimodularity_precision)

    def test_init(self):
        self.assertEqual(self.dbump.center, self.center)
        self.assertEqual(self.dbump.width, self.width)
        self.assertEqual(self.dbump.unimodularity_precision,
                         self.unimodularity_precision)

    def test_compute_area(self):
        # Compare with a known value
        known_area = 0.2219969080840397
        self.assertAlmostEqual(self.dbump._compute_area(), known_area)

    def test_evaluate(self):
        # Compare with known values
        r = np.array([0.0, 0.5, 0.9])
        known_values = np.array([0.0, 0, 0.0])
        np.testing.assert_almost_equal(self.dbump.evaluate(r), known_values)

    def test_str(self):
        self.assertEqual(str(self.dbump), 'Dbump_1D')


class TestTriangular1D(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 10]])
        self.center = 5
        self.width = 2
        self.triangular = Triangular_1D(self.domain, self.center, self.width)

    def test_init(self):
        self.assertEqual(self.triangular.center, self.center)
        self.assertEqual(self.triangular.width, self.width)
        self.assertEqual(self.triangular.domain, self.domain)

    def test_evaluate(self):
        points = np.array([4, 5, 6])
        expected_values = np.array([0, 1, 0])
        actual_values = self.triangular.evaluate(points,
                                                 check_if_in_domain=False)
        np.testing.assert_array_equal(actual_values, expected_values)

    def test_evaluate_with_return_points(self):
        points = np.array([4, 5, 6])
        expected_values = (points, np.array([0, 1, 0]))
        actual_values = self.triangular.evaluate(points, return_points=True)
        np.testing.assert_array_equal(actual_values, expected_values)

    def test_str(self):
        self.assertEqual(str(self.triangular), 'Triangular_1D')


class TestFourier(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 1]])
        self.fourier_sin = Fourier(self.domain, 'sin', 1)
        self.fourier_cos = Fourier(self.domain, 'cos', 1)

    def test_init(self):
        self.assertEqual(self.fourier_sin.type, 'sin')
        self.assertEqual(self.fourier_sin.order, 1)
        self.assertEqual(self.fourier_sin.period, 1)

    def test_evaluate_sin(self):
        r = np.array([0, 0.25, 0.5, 0.75, 1])
        expected = np.sin(2 * np.pi * r) * np.sqrt(2)
        np.testing.assert_almost_equal(self.fourier_sin.evaluate(r), expected)

    def test_evaluate_cos(self):
        r = np.array([0, 0.25, 0.5, 0.75, 1])
        expected = np.cos(2 * np.pi * r) * np.sqrt(2)
        np.testing.assert_almost_equal(self.fourier_cos.evaluate(r), expected)

    def test_str(self):
        self.assertEqual(str(self.fourier_sin), 'Fourier')
        self.assertEqual(str(self.fourier_cos), 'Fourier')


if __name__ == '__main__':
    unittest.main()
