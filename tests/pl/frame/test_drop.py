import pickle
import unittest
from unittest.mock import Mock
from swak.pl import Drop


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.drop = Drop()

    def test_has_columns(self):
        self.assertTrue(hasattr(self.drop, 'columns'))

    def test_columns(self):
        self.assertTupleEqual((), self.drop.columns)

    def test_has_strict(self):
        self.assertTrue(hasattr(self.drop, 'strict'))

    def test_strict(self):
        self.assertIsInstance(self.drop.strict, bool)
        self.assertTrue(self.drop.strict)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.columns = 'foo', 'bar'
        self.strict = False
        self.drop = Drop(*self.columns, strict=self.strict)

    def test_columns(self):
        self.assertTupleEqual(self.columns, self.drop.columns)

    def test_strict(self):
        self.assertIs(self.drop.strict, self.strict)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.columns = 'foo', 'bar'
        self.strict = False
        self.drop = Drop(*self.columns, strict=self.strict)

    def test_callable(self):
        self.assertTrue(callable(self.drop))

    def test_filter_called(self):
        df = Mock()
        _ = self.drop(df)
        df.drop.assert_called_once_with(*self.columns, strict=self.strict)

    def test_return_value(self):
        df = Mock()
        df.drop = Mock(return_value='answer')
        actual = self.drop(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        drop = Drop()
        expected = 'Drop(strict=True)'
        self.assertEqual(expected, repr(drop))

    def test_custom_repr(self):
        drop = Drop('foo', 'bar', strict=False)
        expected = "Drop('foo', 'bar', strict=False)"
        self.assertEqual(expected, repr(drop))

    def test_pickle_works(self):
        drop = Drop('foo', 'bar', strict=False)
        _ = pickle.loads(pickle.dumps(drop))


if __name__ == '__main__':
    unittest.main()
