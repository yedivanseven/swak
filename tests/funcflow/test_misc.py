import os
import unittest
from swak.funcflow import unit, identity, exit_ok


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
        a = 'foo'
        result = identity(a)
        self.assertIs(result, a)

    def test_many(self):
        expected = 1, 'foo', False
        actual = identity(*expected)
        self.assertTupleEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
