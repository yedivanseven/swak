import pickle
import unittest
from unittest.mock import Mock
from swak.pl import Tail


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.tail = Tail()

    def test_has_n(self):
        self.assertTrue(hasattr(self.tail, 'n'))

    def test_n(self):
        self.assertIsInstance(self.tail.n, int)
        self.assertEqual(5, self.tail.n)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.n = 10
        self.tail = Tail(self.n)

    def test_n(self):
        self.assertEqual(self.n, self.tail.n)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        self.assertTrue(callable(Tail()))

    def test_tail_called_default(self):
        df = Mock()
        _ = Tail()(df)
        df.tail.assert_called_once_with(5)

    def test_tail_called(self):
        df = Mock()
        _ = Tail(10)(df)
        df.tail.assert_called_once_with(10)

    def test_return_value(self):
        df = Mock()
        df.tail = Mock(return_value='answer')
        actual = Tail(10)(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        tail = Tail()
        expected = 'Tail(5)'
        self.assertEqual(expected, repr(tail))

    def test_custom_repr(self):
        tail = Tail(10)
        expected = "Tail(10)"
        self.assertEqual(expected, repr(tail))

    def test_pickle_works(self):
        drop = Tail(10)
        _ = pickle.loads(pickle.dumps(drop))


if __name__ == '__main__':
    unittest.main()
