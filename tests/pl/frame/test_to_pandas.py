import pickle
import unittest
from unittest.mock import Mock
from swak.pl import ToPandas


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.to_pandas = ToPandas()

    def test_has_use_pyarrow_extension_array(self):
        self.assertTrue(hasattr(self.to_pandas, 'use_pyarrow_extension_array'))

    def test_use_pyarrow_extension_array(self):
        self.assertIs(self.to_pandas.use_pyarrow_extension_array, False)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.to_pandas, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.to_pandas.kwargs)


class TestCustomAttributes(unittest.TestCase):

    def setUp(self):
        self.use_pyarrow_extension_array = True
        self.kwargs = {'foo': 1, 'bar': 'baz'}
        self.to_pandas = ToPandas(
            use_pyarrow_extension_array=self.use_pyarrow_extension_array,
            **self.kwargs
        )

    def test_use_pyarrow_extension_array(self):
        self.assertIs(self.to_pandas.use_pyarrow_extension_array, True)

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.to_pandas.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.use_pyarrow_extension_array = True
        self.kwargs = {'foo': 1, 'bar': 'baz'}
        self.to_pandas = ToPandas(
            use_pyarrow_extension_array=self.use_pyarrow_extension_array,
            **self.kwargs
        )

    def test_callable(self):
        self.assertTrue(callable(self.to_pandas))

    def test_to_pandas_called(self):
        df = Mock()
        _ = self.to_pandas(df)
        df.to_pandas.assert_called_once_with(
            use_pyarrow_extension_array=self.use_pyarrow_extension_array,
            **self.kwargs
        )

    def test_return_value(self):
        df = Mock()
        df.to_pandas = Mock(return_value='pandas_df')
        actual = self.to_pandas(df)
        self.assertEqual('pandas_df', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        to_pandas = ToPandas()
        expected = "ToPandas(use_pyarrow_extension_array=False)"
        self.assertEqual(expected, repr(to_pandas))

    def test_custom_repr(self):
        to_pandas = ToPandas(use_pyarrow_extension_array=True, foo=1, bar=2)
        expected = "ToPandas(use_pyarrow_extension_array=True, foo=1, bar=2)"
        self.assertEqual(expected, repr(to_pandas))

    def test_pickle_works(self):
        to_pandas = ToPandas(use_pyarrow_extension_array=True, foo=1, bar=2)
        _ = pickle.loads(pickle.dumps(to_pandas))


if __name__ == '__main__':
    unittest.main()
