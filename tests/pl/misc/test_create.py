import pickle
import unittest
from unittest.mock import patch
from swak.pl import Create


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.create = Create()

    def test_has_schema(self):
        self.assertTrue(hasattr(self.create, 'schema'))

    def test_schema(self):
        self.assertIsNone(self.create.schema)

    def test_has_schema_overrides(self):
        self.assertTrue(hasattr(self.create, 'schema_overrides'))

    def test_schema_overrides(self):
        self.assertIsNone(self.create.schema_overrides)

    def test_has_strict(self):
        self.assertTrue(hasattr(self.create, 'strict'))

    def test_strict(self):
        self.assertIsInstance(self.create.strict, bool)
        self.assertTrue(self.create.strict)

    def test_has_orient(self):
        self.assertTrue(hasattr(self.create, 'orient'))

    def test_orient(self):
        self.assertIsNone(self.create.orient)

    def test_has_infer_schema_length(self):
        self.assertTrue(hasattr(self.create, 'infer_schema_length'))

    def test_infer_schema_length(self):
        self.assertIsInstance(self.create.infer_schema_length, int)
        self.assertEqual(100, self.create.infer_schema_length)

    def test_has_nan_to_null(self):
        self.assertTrue(hasattr(self.create, 'nan_to_null'))

    def test_nan_to_null(self):
        self.assertIsInstance(self.create.nan_to_null, bool)
        self.assertFalse(self.create.nan_to_null)

    def test_has_height(self):
        self.assertTrue(hasattr(self.create, 'height'))

    def test_height(self):
        self.assertIsNone(self.create.height)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.create = Create(
            schema=[],
            schema_overrides={},
            strict=False,
            orient='row',
            infer_schema_length=50,
            nan_to_null=True,
            height=100
        )

    def test_schema(self):
        self.assertListEqual([], self.create.schema)

    def test_schema_overrides(self):
        self.assertDictEqual({}, self.create.schema_overrides)

    def test_strict(self):
        self.assertIsInstance(self.create.strict, bool)
        self.assertFalse(self.create.strict)

    def test_orient(self):
        self.assertEqual('row', self.create.orient)

    def infer_schema_length(self):
        self.assertIsInstance(self.create.infer_schema_length, int)
        self.assertEqual(50, self.create.infer_schema_length)

    def test_nan_to_null(self):
        self.assertIsInstance(self.create.nan_to_null, bool)
        self.assertTrue(self.create.nan_to_null)

    def test_height(self):
        self.assertIsInstance(self.create.height, int)
        self.assertEqual(100, self.create.height)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        self.assertTrue(callable(Create()))

    @patch('swak.pl.misc.DataFrame')
    def test_call_default(self, mock):
        obj = object()
        _ = Create()(obj)
        mock.assert_called_once_with(
            obj,
            None,
            schema_overrides=None,
            strict=True,
            orient=None,
            infer_schema_length=100,
            nan_to_null=False,
            height=None
        )

    @patch('swak.pl.misc.DataFrame')
    def test_call(self, mock):
        obj = object()
        _ = Create(
            schema=[],
            schema_overrides={},
            strict=False,
            orient='row',
            infer_schema_length=50,
            nan_to_null=True,
            height=100
        )(
            obj
        )
        mock.assert_called_once_with(
            obj,
            [],
            schema_overrides={},
            strict=False,
            orient='row',
            infer_schema_length=50,
            nan_to_null=True,
            height=100
        )


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        create = Create()
        expected = ('Create(None, schema_overrides=None, strict=True, '
                    'orient=None, infer_schema_length=100, nan_to_null=False,'
                    ' height=None)')
        self.assertEqual(expected, repr(create))

    def test_repr(self):
        create = Create([], {},True, 'row', 50, True, 100)
        expected = ("Create([], schema_overrides={}, strict=True, "
                    "orient='row', infer_schema_length=50, nan_to_null=True, "
                    "height=100)")
        self.assertEqual(expected, repr(create))

    def test_pickle_works(self):
        create = Create()
        _ = pickle.loads(pickle.dumps(create))


if __name__ == '__main__':
    unittest.main()
