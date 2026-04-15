import unittest
import torch as pt
from swak.pt.exceptions import ShapeError, ValidationErrors
from swak.pt.misc import LazyCatDim0


class TestLazyCatDim0(unittest.TestCase):

    def setUp(self):
        two = pt.rand(2, 6)
        three = pt.rand(3, 6)
        self.device = two.device
        self.expected = pt.cat([two, three], dim=0)
        self.cat = LazyCatDim0([two, three])
        two_1d = pt.rand(2)
        three_1d = pt.rand(3)
        self.expected_1d = pt.cat([two_1d, three_1d], dim=0)
        self.cat_1d = LazyCatDim0([two_1d, three_1d])
        two_3d = pt.rand(2, 6, 9)
        three_3d = pt.rand(3, 6, 9)
        self.expected_3d = pt.cat([two_3d, three_3d], dim=0)
        self.cat_3d = LazyCatDim0([two_3d, three_3d])

    def test_empty_iterable_raises(self):
        with self.assertRaises(ShapeError):
            _ = LazyCatDim0([])

    def test_scalar_tensors_raise(self):
        tensors = pt.ones(10), pt.tensor(2.0)
        with self.assertRaises(ValidationErrors):
            _ = LazyCatDim0(tensors)

    def test_different_shapes_raise(self):
        tensors = pt.ones(2, 5), pt.ones(10)
        with self.assertRaises(ValidationErrors):
            _ = LazyCatDim0(tensors)

    @unittest.skipUnless(pt.cuda.is_available(), 'CUDA is not available.')
    def test_different_devices_raise(self):
        tensors = pt.ones(2, 5, device='cpu'), pt.ones(3, 5, device='cuda')
        with self.assertRaises(ValidationErrors):
            _ = LazyCatDim0(tensors)

    def test_different_types_raise(self):
        tensors = pt.ones(2, 5), pt.ones(3, 5, dtype=pt.long)
        with self.assertRaises(ValidationErrors):
            _ = LazyCatDim0(tensors)

    def test_has_lookup(self):
        self.assertTrue(hasattr(self.cat, 'lookup'))

    def test_lookup(self):
        self.assertIsInstance(self.cat.lookup, tuple)
        expected = (0, 0), (0, 1), (1, 0), (1, 1), (1, 2)
        self.assertTupleEqual(expected, self.cat.lookup)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.cat, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.cat.dtype, pt.float)

    def test_has_device(self):
        self.assertTrue(hasattr(self.cat, 'device'))

    def test_device(self):
        self.assertEqual(self.device, self.cat.device)

    def test_has_shape(self):
        self.assertTrue(hasattr(self.cat, 'shape'))

    def test_shape(self):
        expected = pt.Size((5, 6))
        self.assertEqual(expected, self.cat.shape)

    def test_has_size(self):
        self.assertTrue(hasattr(self.cat, 'size'))

    def test_size_callable(self):
        self.assertTrue(callable(self.cat.size))

    def test_call_size(self):
        expected = pt.Size((5, 6))
        self.assertEqual(expected, self.cat.size())

    def test_call_size_arg(self):
        self.assertEqual(5, self.cat.size(0))
        self.assertEqual(6, self.cat.size(1))

    def test_has_to(self):
        self.assertTrue(hasattr(self.cat, 'to'))

    def test_to_callable(self):
        self.assertTrue(callable(self.cat.to))

    def test_call_to(self):
        cast = self.cat.to(pt.long)
        self.assertIs(cast.dtype, pt.long)

    def test_repr(self):
        expected = 'LazyCatDim0(n=5)'
        self.assertEqual(expected, repr(self.cat))

    def test_len(self):
        self.assertEqual(5, len(self.cat))

    def test_iter(self):
        counter = 0
        for tensor in self.cat:
            self.assertEqual(pt.Size([6]), tensor.shape)
            counter += 1
        self.assertEqual(5, counter)

    def test_contains_true(self):
        cat = LazyCatDim0([pt.tensor([1, 2, 3])])
        self.assertTrue(2 in cat)

    def test_contains_false(self):
        cat = LazyCatDim0([pt.tensor([1, 2, 3])])
        self.assertFalse(4 in cat)

    def test_getitem_int(self):
        pt.testing.assert_close(self.cat[0], self.expected[0])
        pt.testing.assert_close(self.cat[1], self.expected[1])
        pt.testing.assert_close(self.cat[2], self.expected[2])
        pt.testing.assert_close(self.cat[3], self.expected[3])
        pt.testing.assert_close(self.cat[4], self.expected[4])

    def test_getitem_slice_step_1(self):
        pt.testing.assert_close(self.cat[1:-1], self.expected[1:-1])

    def test_getitem_slice_step_2(self):
        pt.testing.assert_close(self.cat[1:-1:2], self.expected[1:-1:2])

    def test_getitem_list(self):
        pt.testing.assert_close(self.cat[[1, 2, 4]], self.expected[[1, 2, 4]])

    def test_getitem_empty_tuple(self):
        pt.testing.assert_close(self.cat[()], self.expected[()])

    def test_getitem_one_tuple_int(self):
        pt.testing.assert_close(self.cat[2,], self.expected[2,])

    def test_getitem_one_tuple_slice(self):
        pt.testing.assert_close(self.cat[2:4,], self.expected[2:4,])

    def test_getitem_tuple_int(self):
        pt.testing.assert_close(self.cat[2, 4], self.expected[2, 4])

    def test_getitem_tuple_slice_int(self):
        pt.testing.assert_close(self.cat[1:4, 3], self.expected[1:4, 3])

    def test_getitem_tuple_int_slice(self):
        pt.testing.assert_close(self.cat[3, 1:4], self.expected[3, 1:4])

    def test_getitem_tuple_slice_slice(self):
        pt.testing.assert_close(self.cat[1:3, 2:4], self.expected[1:3, 2:4])

    def test_getitem_int_1d(self):
        pt.testing.assert_close(self.cat_1d[2], self.expected_1d[2])

    def test_getitem_slice_step_1_1d(self):
        pt.testing.assert_close(self.cat_1d[1:-1], self.expected_1d[1:-1])

    def test_getitem_slice_step_2_1d(self):
        pt.testing.assert_close(self.cat_1d[1:-1:2], self.expected_1d[1:-1:2])

    def test_getitem_list_1d(self):
        pt.testing.assert_close(
            self.cat_1d[[1, 2, 4]],
            self.expected_1d[[1, 2, 4]]
        )

    def test_getitem_empty_tuple_1d(self):
        pt.testing.assert_close(self.cat_1d[()], self.expected_1d[()])

    def test_getitem_one_tuple_int_1d(self):
        pt.testing.assert_close(self.cat_1d[2,], self.expected_1d[2,])

    def test_getitem_one_tuple_slice_1d(self):
        pt.testing.assert_close(self.cat_1d[2:4, ], self.expected_1d[2:4, ])

    def test_getitem_int_3d(self):
        pt.testing.assert_close(self.cat_3d[2], self.expected_3d[2])

    def test_getitem_slice_step_1_3d(self):
        pt.testing.assert_close(self.cat_3d[1:-1], self.expected_3d[1:-1])

    def test_getitem_slice_step_2_3d(self):
        pt.testing.assert_close(self.cat_3d[1:-1:2], self.expected_3d[1:-1:2])

    def test_getitem_list_3d(self):
        pt.testing.assert_close(
            self.cat_3d[[1, 2, 4]],
            self.expected_3d[[1, 2, 4]]
        )

    def test_getitem_empty_tuple_3d(self):
        pt.testing.assert_close(self.cat_3d[()], self.expected_3d[()])

    def test_getitem_one_tuple_int_3d(self):
        pt.testing.assert_close(self.cat_3d[2,], self.expected_3d[2,])

    def test_getitem_one_tuple_slice_3d(self):
        pt.testing.assert_close(self.cat_3d[2:4, ], self.expected_3d[2:4, ])

    def test_getitem_tuple_int_3d(self):
        pt.testing.assert_close(self.cat_3d[2, 4], self.expected_3d[2, 4])
        pt.testing.assert_close(
            self.cat_3d[2, 4, 6],
            self.expected_3d[2, 4, 6]
        )

    def test_getitem_tuple_slice_int_3d(self):
        pt.testing.assert_close(self.cat_3d[1:4, 3], self.expected_3d[1:4, 3])
        pt.testing.assert_close(
            self.cat_3d[1:4, 3, 6],
            self.expected_3d[1:4, 3, 6]
        )
        pt.testing.assert_close(
            self.cat_3d[1:4, 3, 2:6],
            self.expected_3d[1:4, 3, 2:6]
        )

    def test_getitem_tuple_int_slice_3d(self):
        pt.testing.assert_close(self.cat_3d[3, 1:4], self.expected_3d[3, 1:4])
        pt.testing.assert_close(
            self.cat_3d[3, 1:4, 6],
            self.expected_3d[3, 1:4, 6]
        )
        pt.testing.assert_close(
            self.cat_3d[3, 1:4, 2:6],
            self.expected_3d[3, 1:4, 2:6]
        )

    def test_getitem_tuple_slice_slice_3d(self):
        pt.testing.assert_close(
            self.cat_3d[1:3, 2:4],
            self.expected_3d[1:3, 2:4]
        )
        pt.testing.assert_close(
            self.cat_3d[1:3, 2:4, 6],
            self.expected_3d[1:3, 2:4, 6]
        )
        pt.testing.assert_close(
            self.cat_3d[1:3, 2:4, 2:6],
            self.expected_3d[1:3, 2:4, 2:6]
        )


if __name__ == '__main__':
    unittest.main()
