import pickle
import unittest
from unittest.mock import patch
from swak.pl import FromPandas


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.from_pd = FromPandas()

    def test_has_schema_overrides(self):
        self.assertTrue(hasattr(self.from_pd, 'schema_overrides'))

    def test_schema_overrides(self):
        self.assertIsNone(self.from_pd.schema_overrides)

    def test_has_has_rechunk(self):
        self.assertTrue(hasattr(self.from_pd, 'rechunk'))

    def test_rechunk(self):
        self.assertIsInstance(self.from_pd.rechunk, bool)
        self.assertTrue(self.from_pd.rechunk)

    def test_has_nan_to_null(self):
        self.assertTrue(hasattr(self.from_pd, 'nan_to_null'))

    def test_nan_to_null(self):
        self.assertIsInstance(self.from_pd.nan_to_null, bool)
        self.assertTrue(self.from_pd.nan_to_null)

    def test_has_include_index(self):
        self.assertTrue(hasattr(self.from_pd, 'include_index'))

    def test_include_index(self):
        self.assertIsInstance(self.from_pd.include_index, bool)
        self.assertFalse(self.from_pd.include_index)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.from_pd = FromPandas(
            schema_overrides={},
            rechunk=False,
            nan_to_null=False,
            include_index=True
        )

    def test_schema_overrides(self):
        self.assertDictEqual({}, self.from_pd.schema_overrides)

    def test_rechunk(self):
        self.assertIsInstance(self.from_pd.rechunk, bool)
        self.assertFalse(self.from_pd.rechunk)

    def test_nan_to_null(self):
        self.assertIsInstance(self.from_pd.nan_to_null, bool)
        self.assertFalse(self.from_pd.nan_to_null)

    def test_include_index(self):
        self.assertIsInstance(self.from_pd.include_index, bool)
        self.assertTrue(self.from_pd.include_index)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        self.assertTrue(callable(FromPandas()))

    @patch('polars.from_pandas')
    def test_call_default(self, mock):
        obj = object()
        _ = FromPandas()(obj)
        mock.assert_called_once_with(
            obj,
            schema_overrides=None,
            rechunk=True,
            nan_to_null=True,
            include_index=False,
        )

    @patch('polars.from_pandas')
    def test_call(self, mock):
        obj = object()
        _ = FromPandas(
                schema_overrides={},
                rechunk=False,
                nan_to_null=False,
                include_index=True
        )(
            obj
        )
        mock.assert_called_once_with(
            obj,
            schema_overrides={},
            rechunk=False,
            nan_to_null=False,
            include_index=True,
        )


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        from_pd = FromPandas()
        expected = ('FromPandas(schema_overrides=None, rechunk=True, '
                    'nan_to_null=True, include_index=False)')
        self.assertEqual(expected, repr(from_pd))

    def test_repr(self):
        from_pd = FromPandas({}, False, False, True)
        expected = ('FromPandas(schema_overrides={}, rechunk=False, '
                    'nan_to_null=False, include_index=True)')
        self.assertEqual(expected, repr(from_pd))

    def test_pickle_works(self):
        from_pd = FromPandas()
        _ = pickle.loads(pickle.dumps(from_pd))


if __name__ == '__main__':
    unittest.main()
