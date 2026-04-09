import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import Copy


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.copy = Copy()

    def test_has_deep(self):
        self.assertTrue(hasattr(self.copy, 'deep'))

    def test_deep(self):
        self.assertIsInstance(self.copy.deep, bool)
        self.assertTrue(self.copy.deep)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.copy = Copy(deep=False)

    def test_deep(self):
        self.assertIsInstance(self.copy.deep, bool)
        self.assertFalse(self.copy.deep)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.copy = Copy(deep=False)

    def test_callable(self):
        self.assertTrue(callable(self.copy))

    def test_copy_called(self):
        df = pd.DataFrame(range(10))
        df.copy = Mock(return_value='expected')
        actual = self.copy(df)
        df.copy.assert_called_once_with(False)
        self.assertEqual('expected', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        copy = Copy()
        expected = 'Copy(True)'
        self.assertEqual(expected, repr(copy))

    def test_custom_repr(self):
        copy = Copy(False)
        expected = 'Copy(False)'
        self.assertEqual(expected, repr(copy))

    def test_pickle_works(self):
        copy = Copy()
        _ = pickle.loads(pickle.dumps(copy))


if __name__ == '__main__':
    unittest.main()
