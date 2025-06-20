import unittest
from unittest.mock import Mock
from swak.funcflow import unit, identity, apply


class TestUnit(unittest.TestCase):

    def test_empty(self):
        actual = unit()
        self.assertIsInstance(actual, tuple)
        self.assertTupleEqual((), actual)

    def test_one(self):
        actual = unit('foo')
        self.assertIsInstance(actual, tuple)
        self.assertTupleEqual((), actual)

    def test_many(self):
        actual = unit(1, 'foo', False)
        self.assertIsInstance(actual, tuple)
        self.assertTupleEqual((), actual)

    def test_empty_tuple(self):
        actual = unit(())
        self.assertTupleEqual((), actual)

    def test_one_tuple(self):
        actual = unit((42,))
        self.assertTupleEqual((), actual)

    def test_two_tuple(self):
        actual = unit((42, 'foo'))
        self.assertTupleEqual((), actual)


class TestIdentity(unittest.TestCase):

    def test_empty(self):
        actual = identity()
        self.assertIsInstance(actual, tuple)
        self.assertTupleEqual((), actual)

    def test_one_arg(self):
        expected = 'foo'
        actual = identity(expected)
        self.assertIs(expected, actual)

    def test_many(self):
        expected = 1, 'foo', False
        actual = identity(*expected)
        self.assertTupleEqual(expected, actual)

    def test_empty_tuple(self):
        expected = ()
        actual = identity(expected)
        self.assertTupleEqual(expected, actual)

    def test_one_tuple(self):
        expected = 42,
        actual = identity(expected)
        self.assertTupleEqual(expected, actual)

    def test_two_tuple(self):
        expected = 42, 'foo'
        actual = identity(expected)
        self.assertTupleEqual(expected, actual)


class TestApply(unittest.TestCase):

    def test_call_called_with_no_args(self):
        mock = Mock()
        _ = apply(mock)
        mock.assert_called_once()
        mock.assert_called_once_with()

    def test_call_called_with_one_arg(self):
        mock = Mock()
        _ = apply(mock, 1)
        mock.assert_called_once()
        mock.assert_called_once_with(1)

    def test_call_called_with_multiple_args(self):
        mock = Mock()
        _ = apply(mock, 1, 'foo', False)
        mock.assert_called_once()
        mock.assert_called_once_with(1, 'foo', False)

    def test_call_called_with_empty_tuple(self):
        mock = Mock()
        _ = apply(mock, ())
        mock.assert_called_once()
        mock.assert_called_once_with(())

    def test_call_called_with_one_tuple(self):
        mock = Mock()
        _ = apply(mock, (42,))
        mock.assert_called_once()
        mock.assert_called_once_with((42,))

    def test_call_called_with_two_tuple(self):
        mock = Mock()
        _ = apply(mock, (42, 'foo'))
        mock.assert_called_once()
        mock.assert_called_once_with((42, 'foo'))


if __name__ == '__main__':
    unittest.main()
