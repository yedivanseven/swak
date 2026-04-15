import pickle
import unittest
from unittest.mock import Mock
from swak.pl import VStack


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.stack = VStack()

    def test_has_in_place(self):
        self.assertTrue(hasattr(self.stack, 'in_place'))

    def test_in_place(self):
        self.assertIsInstance(self.stack.in_place, bool)
        self.assertFalse(self.stack.in_place)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.in_place = True
        self.stack = VStack(self.in_place)

    def test_in_place(self):
        self.assertIs(self.in_place, self.stack.in_place)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.in_place = True
        self.stack = VStack(self.in_place)

    def test_callable(self):
        self.assertTrue(callable(self.stack))

    def test_drop_called(self):
        upper = Mock()
        lower = Mock()
        _ = self.stack(upper, lower)
        upper.vstack.assert_called_once_with(lower, in_place=self.in_place)

    def test_return_value(self):
        upper = Mock()
        lower = Mock()
        upper.vstack = Mock(return_value='answer')
        actual = self.stack(upper, lower)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        stack = VStack()
        expected = "VStack(in_place=False)"
        self.assertEqual(expected, repr(stack))

    def test_custom_repr(self):
        stack = VStack(True)
        expected = "VStack(in_place=True)"
        self.assertEqual(expected, repr(stack))

    def test_pickle_works(self):
        stack = VStack()
        _ = pickle.loads(pickle.dumps(stack))


if __name__ == '__main__':
    unittest.main()
