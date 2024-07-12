import os
import unittest
from unittest.mock import Mock
from swak.funcflow import unit, identity, exit_ok, apply


class TestExitOk(unittest.TestCase):

    def test_empty(self):
        actual = exit_ok()
        self.assertIsInstance(actual, int)
        self.assertEqual(os.EX_OK, actual)

    def test_one(self):
        actual = exit_ok('foo')
        self.assertIsInstance(actual, int)
        self.assertEqual(os.EX_OK, actual)

    def test_many(self):
        actual = exit_ok(1, 'foo', False)
        self.assertIsInstance(actual, int)
        self.assertEqual(os.EX_OK, actual)


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


if __name__ == '__main__':
    unittest.main()
