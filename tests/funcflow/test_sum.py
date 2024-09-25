import unittest
import pickle
from swak.funcflow import Sum
from swak.funcflow.exceptions import SumError
from swak.misc import ArgRepr, IndentRepr


class A(ArgRepr):

    def __init__(self, a):
        super().__init__(a)
        self.a = a


class Ind(IndentRepr):
    pass


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Sum()

    def test_has_acc(self):
        s = Sum()
        self.assertTrue(hasattr(s, 'acc'))

    def test_acc_correct(self):
        s = Sum()
        self.assertIsNone(s.acc)


class TestDefaultUsage(unittest.TestCase):

    def setUp(self):
        self.s = Sum()

    def test_callable(self):
        self.assertTrue(callable(self.s))

    def test_empty_raises(self):
        with self.assertRaises(StopIteration):
            _ = self.s([])

    def test_single_returns_first_element(self):
        actual = self.s([1])
        self.assertIsInstance(actual, int)
        self.assertEqual(1, actual)

    def test_return_correct(self):
        actual = self.s([1, 2, 3])
        self.assertIsInstance(actual, int)
        self.assertEqual(6, actual)

    def test_wrong_iterator_raises(self):
        with self.assertRaises(TypeError):
            _ = self.s(1)

    def test_wrong_call_raises(self):
        expected = ("Error adding element #2:\n"
                    "foo\n"
                    "TypeError:\n"
                    "unsupported operand type(s) for +: 'int' and 'str'")
        with self.assertRaises(SumError) as error:
            _ = self.s([1, 2, 'foo'])
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_argrepr(self):
        expected = ("Error adding element #2:\n"
                    "A(1)\n"
                    "TypeError:\n"
                    "unsupported operand type(s) for +: 'int' and 'A'")
        with self.assertRaises(SumError) as error:
            _ = self.s([1, 2, A(1)])
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_indentrepr(self):
        expected = ("Error adding element #2:\n"
                    "Ind():\n"
                    "[ 0] 1\n"
                    "TypeError:\n"
                    "unsupported operand type(s) for +: 'int' and 'Ind'")
        with self.assertRaises(SumError) as error:
            _ = self.s([1, 2, Ind([1])])
        self.assertEqual(expected, str(error.exception))


class TestAccAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Sum(1)

    def test_has_acc(self):
        s = Sum(1)
        self.assertTrue(hasattr(s, 'acc'))

    def test_acc_correct(self):
        s = Sum(1)
        self.assertIsInstance(s.acc, int)
        self.assertEqual(1, s.acc)


class TestAccUsage(unittest.TestCase):

    def setUp(self):
        self.s = Sum(1)

    def test_callable(self):
        self.assertTrue(callable(self.s))

    def test_empty_returns_acc(self):
        actual = self.s([])
        self.assertIsInstance(actual, int)
        self.assertEqual(1, actual)

    def test_return_correct(self):
        actual = self.s([2, 3])
        self.assertIsInstance(actual, int)
        self.assertEqual(6, actual)

    def test_wrong_iterator_raises(self):
        with self.assertRaises(TypeError):
            _ = self.s(1)

    def test_wrong_call_raises(self):
        expected = ("Error adding element #2:\n"
                    "foo\n"
                    "TypeError:\n"
                    "unsupported operand type(s) for +: 'int' and 'str'")
        with self.assertRaises(SumError) as error:
            _ = self.s([1, 2, 'foo'])
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_argrepr(self):
        expected = ("Error adding element #2:\n"
                    "A(1)\n"
                    "TypeError:\n"
                    "unsupported operand type(s) for +: 'int' and 'A'")
        with self.assertRaises(SumError) as error:
            _ = self.s([1, 2, A(1)])
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_indentrepr(self):
        expected = ("Error adding element #2:\n"
                    "Ind():\n"
                    "[ 0] 1\n"
                    "TypeError:\n"
                    "unsupported operand type(s) for +: 'int' and 'Ind'")
        with self.assertRaises(SumError) as error:
            _ = self.s([1, 2, Ind([1])])
        self.assertEqual(expected, str(error.exception))


class TestMisc(unittest.TestCase):

    def test_default_pickle_works(self):
        s = Sum()
        _ = pickle.dumps(s)

    def test_acc_pickle_works(self):
        s = Sum(1)
        _ = pickle.dumps(s)

    def test_default_repr(self):
        s = Sum()
        self.assertEqual('Sum(None)', repr(s))

    def test_acc_repr(self):
        s = Sum(1)
        self.assertEqual('Sum(1)', repr(s))

    def test_acc_argrepr(self):
        s = Sum(A(1))
        self.assertEqual('Sum(A(1))', repr(s))

    def test_acc_indentrepr(self):
        s = Sum(Ind([1, 2, 3]))
        self.assertEqual('Sum(Ind()[3])', repr(s))

    def test_type_annotation(self):
        _ = Sum[int, float]()

    def test_type_annotation_acc(self):
        _ = Sum[int, bool](1)


if __name__ == '__main__':
    unittest.main()
