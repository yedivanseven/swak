import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Reduce
from swak.magic import ArgRepr, IndentRepr


def f(x, y):
    return x + y


class TestInstantiation(unittest.TestCase):

    def test_default(self):
        red = Reduce(f)
        self.assertTrue(hasattr(red, 'call'))
        self.assertIs(red.call, f)
        self.assertTrue(hasattr(red, 'acc'))
        self.assertIsNone(red.acc)

    def test_acc(self):
        red = Reduce(f, ())
        self.assertTrue(hasattr(red, 'acc'))
        self.assertIsInstance(red.acc, tuple)
        self.assertTupleEqual((), red.acc)

    def test_pickle_works(self):
        red = Reduce(f)
        _ = pickle.dumps(red)

    def test_pickle_raises_lambda(self):
        red = Reduce(lambda x, y: x + y)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(red)


class TestCall(unittest.TestCase):

    def test_callable(self):
        red = Reduce(f)
        self.assertTrue(callable(red))

    def test_call_called_no_acc(self):
        mock = Mock(return_value=3)
        red = Reduce[int, int](mock)
        _ = red([1, 2, 4])
        self.assertEqual(2, mock.call_count)
        (a, _), (b, _) = mock.call_args_list
        self.assertTupleEqual((1, 2), a)
        self.assertTupleEqual((3, 4), b)

    def test_call_called_acc(self):
        mock = Mock(return_value=3)
        red = Reduce[int, int](mock, 1)
        _ = red([2, 4, 5])
        self.assertEqual(3, mock.call_count)
        (a, _), (b, _), (c, _) = mock.call_args_list
        self.assertTupleEqual((1, 2), a)
        self.assertTupleEqual((3, 4), b)
        self.assertTupleEqual((3, 5), c)

    def test_return_value_no_acc(self):
        red = Reduce[int, int](f)
        value = red([1, 2, 3])
        self.assertIsInstance(value, int)
        self.assertEqual(6, value)

    def test_return_value_acc(self):
        red = Reduce[tuple, tuple](f)
        value = red([(1,), (2,), (3, )])
        self.assertIsInstance(value, tuple)
        self.assertTupleEqual((1, 2, 3), value)


class TestRepresentation(unittest.TestCase):

    def test_lambda(self):
        red = Reduce(lambda x: x)
        self.assertEqual('Reduce(<lambda>, None)', repr(red))

    def test_function(self):
        red = Reduce(f)
        self.assertEqual('Reduce(f, None)', repr(red))

    def test_class(self):

        class C:
            pass

        red = Reduce(C)
        self.assertEqual('Reduce(C, None)', repr(red))

    def test_argrepr(self):

        class C(ArgRepr):

            def __init__(self, a):
                super().__init__(a)

            def __call__(self):
                pass

        red = Reduce(C('foo'))
        self.assertEqual("Reduce(C('foo'), None)", repr(red))

    def test_indentrepr(self):
        class C(IndentRepr):

            def __init__(self, *xs):
                super().__init__(*xs)

            def __call__(self):
                pass

        red = Reduce(C(1, 2, 3))
        self.assertEqual("Reduce(C, None)", repr(red))

    def test_type(self):
        red = Reduce[int, str](lambda x: x)
        self.assertEqual('Reduce[int, str](<lambda>, None)', repr(red))

    def test_acc(self):
        red = Reduce[int, str](lambda x: x, tuple)
        self.assertEqual('Reduce[int, str](<lambda>, tuple)', repr(red))


if __name__ == '__main__':
    unittest.main()
