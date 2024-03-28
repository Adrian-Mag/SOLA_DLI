import unittest
import numpy as np
from sola.main_classes.domains import HyperParalelipiped


class TestDomain(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 1]])

    def test_sample_domain(self):
        samples = self.domain.sample_domain(10)
        self.assertEqual(samples.shape, (10,))
        self.assertTrue(np.all((0 <= samples) & (samples <= 1)))

    def test_check_if_in_domain(self):
        self.assertTrue(self.domain.check_if_in_domain(0.5))
        self.assertFalse(self.domain.check_if_in_domain(1.5))


if __name__ == '__main__':
    unittest.main()
