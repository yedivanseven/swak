import pickle
import unittest
from unittest.mock import Mock
import polars as pl
from swak.pl import Cast


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.cast = Cast(pl.Int16)

    def test_has_dtypes(self):
        self.assertTrue(hasattr(self.cast, 'dtypes'))

    def test_dtypes(self):
        self.assertIs(self.cast.dtypes, pl.Int16)

    def test_has_strict(self):
        self.assertTrue(hasattr(self.cast, 'strict'))

    def test_strict(self):
        self.assertIsInstance(self.cast.strict, bool)
        self.assertTrue(self.cast.strict)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.dtypes = {'foo': int, 'bar': pl.Int16}
        self.strict = False
        self.cast = Cast(self.dtypes, strict=self.strict)

    def dtypes(self):
        self.assertDictEqual(self.dtypes, self.cast.dtypes)

    def test_strict(self):
        self.assertIs(self.cast.strict, self.strict)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.dtypes = {'foo': int, 'bar': pl.Int16}
        self.strict = False
        self.cast = Cast(self.dtypes, strict=self.strict)

    def test_callable(self):
        self.assertTrue(callable(self.cast))

    def test_cast_called(self):
        df = Mock()
        _ = self.cast(df)
        df.cast.assert_called_once_with(self.dtypes, strict=self.strict)

    def test_return_value(self):
        df = Mock()
        df.cast = Mock(return_value='answer')
        actual = self.cast(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        cast = Cast(pl.Int16)
        expected = 'Cast(Int16, strict=True)'
        self.assertEqual(expected, repr(cast))

    def test_custom_repr(self):
        dtypes = {'foo': int, 'bar': pl.Int16}
        cast = Cast(dtypes, strict=False)
        expected = "Cast({'foo': <class 'int'>, 'bar': Int16}, strict=False)"
        self.assertEqual(expected, repr(cast))

    def test_pickle_works(self):
        drop = Cast(pl.Int16)
        _ = pickle.loads(pickle.dumps(drop))


if __name__ == '__main__':
    unittest.main()
