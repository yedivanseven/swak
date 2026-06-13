import pickle
import unittest
from unittest.mock import patch
from swak.pl import Concat


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.concat = Concat()

    def test_has_how(self):
        self.assertTrue(hasattr(self.concat, 'how'))

    def test_how(self):
        self.assertEqual('vertical', self.concat.how)

    def test_has_has_rechunk(self):
        self.assertTrue(hasattr(self.concat, 'rechunk'))

    def test_rechunk(self):
        self.assertIsInstance(self.concat.rechunk, bool)
        self.assertFalse(self.concat.rechunk)

    def test_has_parallel(self):
        self.assertTrue(hasattr(self.concat, 'parallel'))

    def test_parallel(self):
        self.assertIsInstance(self.concat.parallel, bool)
        self.assertTrue(self.concat.parallel)

    def test_has_strict(self):
        self.assertTrue(hasattr(self.concat, 'strict'))

    def test_strict(self):
        self.assertIsInstance(self.concat.strict, bool)
        self.assertFalse(self.concat.strict)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.concat = Concat(
            'horizontal',
            rechunk=True,
            parallel=False,
            strict=True
        )

    def test_how(self):
        self.assertEqual('horizontal', self.concat.how)

    def test_rechunk(self):
        self.assertIsInstance(self.concat.rechunk, bool)
        self.assertTrue(self.concat.rechunk)

    def test_parallel(self):
        self.assertIsInstance(self.concat.parallel, bool)
        self.assertFalse(self.concat.parallel)

    def test_strict(self):
        self.assertIsInstance(self.concat.strict, bool)
        self.assertTrue(self.concat.strict)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        self.assertTrue(callable(Concat()))

    @patch('polars.concat')
    def test_call_default(self, mock):
        obj = object()
        _ = Concat()(obj)
        mock.assert_called_once_with(
            obj,
            how='vertical',
            rechunk=False,
            parallel=True,
            strict=False
        )

    @patch('polars.concat')
    def test_call(self, mock):
        obj = object()
        _ = Concat(
            how='horizontal',
            rechunk=True,
            parallel=False,
            strict=True
        )(
            obj
        )
        mock.assert_called_once_with(
            obj,
            how='horizontal',
            rechunk=True,
            parallel=False,
            strict=True
        )


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        concat = Concat()
        expected = ("Concat(how='vertical', rechunk=False, "
                    "parallel=True, strict=False)")
        self.assertEqual(expected, repr(concat))

    def test_repr(self):
        concat = Concat('horizontal', True, False, True)
        expected = ("Concat(how='horizontal', rechunk=True, "
                    "parallel=False, strict=True)")
        self.assertEqual(expected, repr(concat))

    def test_pickle_works(self):
        concat = Concat()
        _ = pickle.loads(pickle.dumps(concat))


if __name__ == '__main__':
    unittest.main()
