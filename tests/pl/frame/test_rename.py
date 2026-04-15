import pickle
import unittest
from unittest.mock import Mock
from swak.pl import Rename


def custom(arg):
    return arg * 2


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.map = {'foo': 'bar'}
        self.rename = Rename(self.map)

    def test_has_mapping(self):
        self.assertTrue(hasattr(self.rename, 'mapping'))

    def test_mapping(self):
        self.assertDictEqual(self.map, self.rename.mapping)

    def test_has_strict(self):
        self.assertTrue(hasattr(self.rename, 'strict'))

    def test_strict(self):
        self.assertIsInstance(self.rename.strict, bool)
        self.assertTrue(self.rename.strict)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.strict = True
        self.rename = Rename(custom, strict=self.strict)

    def test_mapping(self):
        self.assertIs(self.rename.mapping, custom)

    def test_strict(self):
        self.assertIs(self.strict, self.rename.strict)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.strict = False
        self.rename = Rename(custom, strict=self.strict)

    def test_callable(self):
        self.assertTrue(callable(self.rename))

    def test_rename_called(self):
        df = Mock()
        _ = self.rename(df)
        df.rename.assert_called_once_with(custom, strict=self.strict)

    def test_return_value(self):
        df = Mock()
        df.rename = Mock(return_value='answer')
        actual = self.rename(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        rename = Rename({'foo': 'bar'})
        expected = "Rename({'foo': 'bar'}, strict=True)"
        self.assertEqual(expected, repr(rename))

    def test_custom_repr(self):
        rename = Rename(custom, strict=False)
        expected = "Rename(custom, strict=False)"
        self.assertEqual(expected, repr(rename))

    def test_pickle_works(self):
        rename = Rename(custom, strict=False)
        _ = pickle.loads(pickle.dumps(rename))

    def test_pickle_raises_on_lambda(self):
        rename = Rename(lambda x: 2*x, strict=False)
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(rename))


if __name__ == '__main__':
    unittest.main()
