import unittest
from unittest.mock import Mock
from swak.jsonobject.fields import Maybe


class TestMaybe(unittest.TestCase):

    def test_instantiation(self):
        _ = Maybe(int)

    def test_type_annotation(self):
        _ = Maybe[int](int)

    def test_has_cast(self):
        maybe = Maybe(int)
        self.assertTrue(hasattr(maybe, 'cast'))

    def test_cast(self):
        maybe = Maybe(int)
        self.assertIs(maybe.cast, int)

    def test_callable(self):
        maybe = Maybe(int)
        self.assertTrue(callable(maybe))

    def test_cast_called_on_not_none(self):
        mock = Mock()
        maybe = Maybe(mock)
        _ = maybe(1)
        mock.assert_called_once()
        mock.assert_called_once_with(1)

    def test_cast_not_called_on_none(self):
        mock = Mock()
        maybe = Maybe(mock)
        _ = maybe(None)
        mock.assert_not_called()

    def test_return_value_not_none(self):
        maybe = Maybe(int)
        result = maybe(1)
        self.assertIsInstance(result, int)
        self.assertEqual(1, result)

    def test_return_value_none(self):
        maybe = Maybe(int)
        result = maybe(None)
        self.assertIsNone(result)

    def test_return_value_null_str(self):
        maybe = Maybe(int)
        result = maybe('null')
        self.assertIsNone(result)

    def test_return_value_none_str(self):
        maybe = Maybe(int)
        result = maybe('None')
        self.assertIsNone(result)

    def test_raises(self):
        maybe = Maybe(int)
        with self.assertRaises(ValueError):
            _ = maybe('1.0')


if __name__ == '__main__':
    unittest.main()
