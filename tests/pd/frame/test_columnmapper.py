import pickle
import unittest
from unittest.mock import MagicMock
import pandas as pd
from swak.pd import ColumnMapper


def identity(x):
    return x


def pow2(x):
    return x**2


class TestDefaultAttributes(unittest.TestCase):

    def test_source(self):
        transform = ColumnMapper('source', identity)
        self.assertTrue(hasattr(transform, 'src_col'))
        self.assertEqual('source', transform.src_col)

    def test_transform(self):
        transform = ColumnMapper('source', identity)
        self.assertTrue(hasattr(transform, 'transform'))
        self.assertIs(identity, transform.transform)

    def test_target(self):
        transform = ColumnMapper('source', identity)
        self.assertTrue(hasattr(transform, 'tgt_col'))
        self.assertEqual(transform.src_col, transform.tgt_col)

    def test_kwargs(self):
        transform = ColumnMapper('source', identity)
        self.assertTrue(hasattr(transform, 'kwargs'))
        self.assertDictEqual({}, transform.kwargs)


class TestCustomAttributes(unittest.TestCase):

    def test_target(self):
        transform = ColumnMapper('source', identity, 'target')
        self.assertTrue(hasattr(transform, 'src_col'))
        self.assertEqual('source', transform.src_col)
        self.assertTrue(hasattr(transform, 'transform'))
        self.assertIs(identity, transform.transform)
        self.assertTrue(hasattr(transform, 'tgt_col'))
        self.assertEqual('target', transform.tgt_col)

    def test_kwargs(self):
        transform = ColumnMapper('source', identity, hello='world')
        self.assertTrue(hasattr(transform, 'src_col'))
        self.assertEqual('source', transform.src_col)
        self.assertTrue(hasattr(transform, 'transform'))
        self.assertIs(identity, transform.transform)
        self.assertTrue(hasattr(transform, 'tgt_col'))
        self.assertEqual('source', transform.tgt_col)
        self.assertTrue(hasattr(transform, 'kwargs'))
        self.assertDictEqual({'hello': 'world'}, transform.kwargs)

    def test_target_and_kwargs(self):
        transform = ColumnMapper('source', identity, 'target', hello='world')
        self.assertTrue(hasattr(transform, 'src_col'))
        self.assertEqual('source', transform.src_col)
        self.assertTrue(hasattr(transform, 'transform'))
        self.assertIs(identity, transform.transform)
        self.assertTrue(hasattr(transform, 'tgt_col'))
        self.assertEqual('target', transform.tgt_col)
        self.assertTrue(hasattr(transform, 'kwargs'))
        self.assertDictEqual({'hello': 'world'}, transform.kwargs)


class TestCall(unittest.TestCase):

    def setUp(self) -> None:
        self.mock_frame = MagicMock()
        self.mock_series = MagicMock()
        self.mock_frame.__getitem__ = MagicMock(return_value=self.mock_series)
        self.df = pd.DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]])
        self.empty = pd.DataFrame(columns=[0, 1])

    def test_callable(self):
        transform = ColumnMapper(0, identity)
        self.assertTrue(callable(transform))

    def test_getitem(self):
        transform = ColumnMapper(0, identity)
        _ = transform(self.mock_frame)
        self.mock_frame.__getitem__.assert_called_once()
        self.mock_frame.__getitem__.assert_called_once_with(0)

    def test_map(self):
        transform = ColumnMapper(0, identity)
        _ = transform(self.mock_frame)
        self.mock_series.map.assert_called_once()
        self.mock_series.map.assert_called_once_with(identity)

    def test_map_kwargs(self):
        transform = ColumnMapper(0, identity, foo='bar')
        _ = transform(self.mock_frame)
        self.mock_series.map.assert_called_once()
        self.mock_series.map.assert_called_once_with(identity, foo='bar')

    def test_replace_return_type(self):
        transform = ColumnMapper(1, pow2)
        transformed = transform(self.df)
        self.assertIsInstance(transformed, pd.DataFrame)

    def test_replace_return_shape(self):
        transform = ColumnMapper(1, pow2)
        transformed = transform(self.df)
        self.assertTupleEqual((4, 2), transformed.shape)

    def test_replace_return_value(self):
        transform = ColumnMapper(1, pow2)
        transformed = transform(self.df)
        pd.testing.assert_series_equal(self.df[1], transformed[1])

    def test_new_col_return_type(self):
        transform = ColumnMapper(1, pow2, 2)
        transformed = transform(self.df)
        self.assertIsInstance(transformed, pd.DataFrame)

    def test_new_col_return_shape(self):
        transform = ColumnMapper(1, pow2, 2)
        transformed = transform(self.df)
        self.assertTupleEqual((4, 3), transformed.shape)

    def test_new_col_return_value(self):
        transform = ColumnMapper(1, pow2, 2)
        transformed = transform(self.df)
        pd.testing.assert_series_equal(self.df[2], transformed[2])

    def test_empty_replace(self):
        transform = ColumnMapper(1, pow2)
        transformed = transform(self.empty)
        self.assertTrue(transformed.empty)
        self.assertTupleEqual((0, 2), transformed.shape)
        pd.testing.assert_frame_equal(self.empty, transformed)

    def test_empty_new_col(self):
        transform = ColumnMapper(1, pow2, 2)
        transformed = transform(self.empty)
        self.assertTrue(transformed.empty)
        self.assertTupleEqual((0, 3), transformed.shape)
        pd.testing.assert_frame_equal(self.empty, transformed)


class TestMisc(unittest.TestCase):

    def test_repr_function(self):
        transform = ColumnMapper(1, pow2)
        expected = "ColumnMapper(1, pow2, None)"
        self.assertEqual(expected, repr(transform))

    def test_repr_dict(self):
        transform = ColumnMapper(1, {3: 4}, 'foo')
        expected = "ColumnMapper(1, dict, 'foo')"
        self.assertEqual(expected, repr(transform))

    def test_repr_series(self):
        transform = ColumnMapper(1, pd.Series({3: 4}), 2)
        expected = "ColumnMapper(1, Series, 2)"
        self.assertEqual(expected, repr(transform))

    def test_pickle_works_with_function(self):
        transform = ColumnMapper(1, pow2)
        _ = pickle.loads(pickle.dumps(transform))

    def test_pickle_raises_with_lambda(self):
        transform = ColumnMapper(1, lambda x: x**2)
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(transform))


if __name__ == '__main__':
    unittest.main()
