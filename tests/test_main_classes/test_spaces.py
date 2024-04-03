import unittest
import numpy as np
from sola.main_classes.spaces import RN, PCb, DirectSumSpace
from sola.main_classes.domains import HyperParalelipiped
from sola.main_classes import functions
from sola.main_classes.functions import Boxcar_1D


class TestRN(unittest.TestCase):
    def setUp(self):
        self.space = RN(3)

    def test_init(self):
        self.assertEqual(self.space.dimension, 3)
        self.assertEqual(self.space.members, {})

    def test_check_if_member(self):
        self.assertTrue(self.space.check_if_member(np.array([[1], [2], [3]])))
        self.assertFalse(self.space.check_if_member(np.array([1, 2, 3, 4])))
        self.assertFalse(self.space.check_if_member(np.array([[1, 2],
                                                              [3, 4]])))
        self.assertFalse(self.space.check_if_member(1))
        self.assertFalse(self.space.check_if_member('not a member'))
        self.space1 = RN(1)
        self.assertTrue(self.space1.check_if_member(1))
        self.assertTrue(self.space1.check_if_member(1.0))
        self.assertTrue(self.space1.check_if_member(np.array([[1]])))
        self.assertTrue(self.space1.check_if_member(np.array([[1.0]])))
        self.assertFalse(self.space1.check_if_member(np.array([1])))
        self.assertFalse(self.space.check_if_member('not a member'))

    def test_random_member(self):
        member = self.space.random_member()
        self.assertEqual(member.shape, (3, 1))
        self.assertTrue(self.space.check_if_member(member))

    def test_add_member(self):
        self.space.add_member('v', np.array([[1], [2], [3]]))
        self.assertIn('v', self.space.members)
        self.assertTrue(np.array_equal(self.space.members['v'],
                                       np.array([[1], [2], [3]])))
        with self.assertRaises(Exception):
            self.space.add_member('w', np.array([[1], [2], [3], [4]]))

    def test_inner_product(self):
        self.assertEqual(self.space.inner_product(np.array([[1], [2], [3]]),
                                                  np.array([[4], [5], [6]])), 32.0) # noqa
        with self.assertRaises(Exception):
            self.space.inner_product('v', 'not a member')

    def test_norm(self):
        self.assertAlmostEqual(self.space.norm(np.array([[1], [2], [3]])),
                               3.7416573867739413)
        with self.assertRaises(Exception):
            self.space.norm('not a member')

    def test_zero(self):
        zero = self.space.zero
        self.assertEqual(zero.shape, (3, 1))
        self.assertTrue(np.array_equal(zero, np.zeros((3, 1))))


class TestPCb(unittest.TestCase):
    def setUp(self):
        self.domain = HyperParalelipiped([[0, 1]])
        self.pcb = PCb(self.domain)

    def test_init(self):
        self.assertEqual(self.pcb.domain, self.domain)
        self.assertEqual(self.pcb.members, {})

    def test_add_member(self):
        function = functions.Gaussian_1D(domain=self.domain,
                                         center=0.5, width=0.1)
        self.pcb.add_member('f', function)
        self.assertEqual(self.pcb.members['f'], function)

    def test_add_member_exception(self):
        with self.assertRaises(Exception):
            self.pcb.add_member('f', 'not a function')

    def test_inner_product(self):
        function1 = functions.Boxcar_1D(domain=self.domain, center=0.4,
                                        width=0.1)
        function2 = functions.Boxcar_1D(domain=self.domain, center=0.5,
                                        width=0.1)
        self.assertEqual(self.pcb.inner_product(function1, function2), 0.0)

    def test_norm(self):
        function = functions.Boxcar_1D(domain=self.domain, center=0.5,
                                       width=0.1)
        self.assertEqual(self.pcb.norm(function), np.sqrt(10))

    def test_zero(self):
        zero_function = self.pcb.zero
        self.assertIsInstance(zero_function, functions.Null_1D)
        self.assertEqual(zero_function.domain, self.domain)


class TestDirectSumSpace(unittest.TestCase):
    def setUp(self):
        domain = HyperParalelipiped([[0, 1]])
        self.space1 = PCb(domain)
        self.space2 = RN(1)
        self.direct_sum_space = DirectSumSpace((self.space1, self.space2))

    def test_init(self):
        self.assertEqual(self.direct_sum_space.spaces,
                         (self.space1, self.space2))

    def test_random_member(self):
        random_member = self.direct_sum_space.random_member()
        self.assertTrue(self.space1.check_if_member(random_member[0]))
        self.assertTrue(self.space2.check_if_member(random_member[1]))
        self.assertFalse(self.space1.check_if_member(random_member[1]))
        self.assertFalse(self.space2.check_if_member(random_member[0]))

    def test_inner_product(self):
        member1_space1 = Boxcar_1D(domain=self.space1.domain, center=0.5,
                                   width=0.1)
        member2_space1 = Boxcar_1D(domain=self.space1.domain, center=0.6,
                                   width=0.1)
        member1_space2 = np.array([[1]])
        member2_space2 = np.array([[2]])
        self.assertEqual(self.direct_sum_space.inner_product((member1_space1,
                                                             member1_space2),
                                                             (member2_space1,
                                                             member2_space2)), 2) # noqa

    def test_norm(self):
        member_space1 = Boxcar_1D(domain=self.space1.domain, center=0.5,
                                  width=0.1)
        member_space2 = np.array([[1]])
        self.assertEqual(self.direct_sum_space.norm((member_space1, member_space2), np.sqrt(10) + 1)) # noqa

    def test_zero(self):
        # Assuming Space1.zero and Space2.zero return 0 and 0 respectively
        self.assertEqual(self.direct_sum_space.zero, (self.space1.zero,
                                                      self.space2.zero))


if __name__ == '__main__':
    unittest.main()
