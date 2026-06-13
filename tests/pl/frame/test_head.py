import pickle
import unittest
from unittest.mock import Mock
from swak.pl import Head


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.head = Head()

    def test_has_n(self):
        self.assertTrue(hasattr(self.head, 'n'))

    def test_n(self):
        self.assertIsInstance(self.head.n, int)
        self.assertEqual(5, self.head.n)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.n = 10
        self.head = Head(self.n)

    def test_n(self):
        self.assertEqual(self.n, self.head.n)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        self.assertTrue(callable(Head()))

    def test_head_called_default(self):
        df = Mock()
        _ = Head()(df)
        df.head.assert_called_once_with(5)

    def test_head_called(self):
        df = Mock()
        _ = Head(10)(df)
        df.head.assert_called_once_with(10)

    def test_return_value(self):
        df = Mock()
        df.head = Mock(return_value='answer')
        actual = Head(10)(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        head = Head()
        expected = 'Head(5)'
        self.assertEqual(expected, repr(head))

    def test_custom_repr(self):
        head = Head(10)
        expected = "Head(10)"
        self.assertEqual(expected, repr(head))

    def test_pickle_works(self):
        drop = Head(10)
        _ = pickle.loads(pickle.dumps(drop))


if __name__ == '__main__':
    unittest.main()
