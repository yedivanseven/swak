import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import FillNA


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.value = 'foo'
        self.fillna = FillNA(self.value)

    def test_has_value(self):
        self.assertTrue(hasattr(self.fillna, 'value'))

    def test_value(self):
        self.assertEqual(self.value, self.fillna.value)

    def test_has_axis(self):
        self.assertTrue(hasattr(self.fillna, 'axis'))

    def test_axis(self):
        self.assertIsInstance(self.fillna.axis, int)
        self.assertEqual(0, self.fillna.axis)

    def test_has_limit(self):
        self.assertTrue(hasattr(self.fillna, 'limit'))

    def test_limit(self):
        self.assertIsNone(self.fillna.limit)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.axis = 'rows'
        self.limit = 42
        self.fillna = FillNA('foo', self.axis, self.limit)

    def test_axis(self):
        self.assertEqual(self.axis, self.fillna.axis)

    def test_limit(self):
        self.assertIsInstance(self.fillna.limit, int)
        self.assertEqual(self.limit, self.fillna.limit)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.value = 'foo'
        self.axis = 'rows'
        self.limit = 42
        self.fillna = FillNA(self.value, self.axis, self.limit)

    def test_fillna_called(self):
        df = pd.DataFrame(range(10))
        df.fillna = Mock(return_value='expected')
        actual = self.fillna(df)
        df.fillna.assert_called_once_with(
            self.value,
            axis=self.axis,
            inplace=False,
            limit=self.limit
        )
        self.assertEqual('expected', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        fillna = FillNA(42)
        expected = 'FillNA(42, axis=0, limit=None)'
        self.assertEqual(expected, repr(fillna))

    def test_cusomt_repr(self):
        fillna = FillNA('foo', 'rows', 42)
        expected = "FillNA('foo', axis='rows', limit=42)"
        self.assertEqual(expected, repr(fillna))

    def test_pickle_works(self):
        fillna = FillNA('foo', 'rows', 42)
        _ = pickle.loads(pickle.dumps(fillna))


if __name__ == '__main__':
    unittest.main()
