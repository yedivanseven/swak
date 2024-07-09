import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Filter
from swak.funcflow.exceptions import FilterError


def greater_3(x: int) -> bool:
    return x > 3


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Filter()

    def test_has_criterion(self):
        f = Filter()
        self.assertTrue(hasattr(f, 'criterion'))

    def test_criterion_is_none(self):
        f = Filter()
        self.assertIsNone(f.criterion)

    def test_has_wrapper(self):
        f = Filter()
        self.assertTrue(hasattr(f, 'wrapper'))

    def test_wrapper_is_none(self):
        f = Filter()
        self.assertIsNone(f.wrapper)


class TestDefaultUsage(unittest.TestCase):

    def setUp(self):
        self.f = Filter()

    def test_callable(self):
        self.assertTrue(callable(self.f))

    def test_empty_list(self):
        actual = self.f([])
        self.assertListEqual([], actual)

    def test_empty_tuple(self):
        actual = self.f(tuple())
        self.assertTupleEqual(tuple(), actual)

    def test_empty_set(self):
        actual = self.f(set())
        self.assertSetEqual(set(), actual)

    def test_list(self):
        actual = self.f([1, 2, 0, 2, 1])
        self.assertListEqual([1, 2, 2, 1], actual)

    def test_tuple(self):
        actual = self.f((1, 2, 0, 2, 1))
        self.assertTupleEqual((1, 2, 2, 1), actual)

    def test_set(self):
        actual = self.f({1, 2, 0, 2, 1})
        self.assertSetEqual({1, 2}, actual)

    def test_wrong_iterable_raises(self):
        expected = ('Could not wrap filter results '
                    'into an instance of list_iterator!')
        with self.assertRaises(FilterError) as error:
            _ = self.f(iter([1, 2, 0, 2, 1]))
        self.assertEqual(expected, str(error.exception))


class TestCriterionAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Filter(greater_3)

    def test_has_criterion(self):
        f = Filter(greater_3)
        self.assertTrue(hasattr(f, 'criterion'))

    def test_criterion_is_none(self):
        f = Filter(greater_3)
        self.assertIs(f.criterion, greater_3)

    def test_has_wrapper(self):
        f = Filter(greater_3)
        self.assertTrue(hasattr(f, 'wrapper'))

    def test_wrapper_is_none(self):
        f = Filter(greater_3)
        self.assertIsNone(f.wrapper)


class TestCriterionUsage(unittest.TestCase):

    def setUp(self):
        self.f = Filter(greater_3)

    def test_callable(self):
        self.assertTrue(callable(self.f))

    def test_empty_list(self):
        actual = self.f([])
        self.assertListEqual([], actual)

    def test_empty_tuple(self):
        actual = self.f(tuple())
        self.assertTupleEqual(tuple(), actual)

    def test_empty_set(self):
        actual = self.f(set())
        self.assertSetEqual(set(), actual)

    def test_criterion_called(self):
        mock = Mock()
        f = Filter(mock)
        _ = f([1])
        mock.assert_called_once()

    def test_criterion_called_correctly(self):
        mock = Mock()
        f = Filter(mock)
        _ = f([1])
        mock.assert_called_once_with(1)

    def test_list(self):
        actual = self.f([1, 2, 3, 4, 5])
        self.assertListEqual([4, 5], actual)

    def test_tuple(self):
        actual = self.f((1, 2, 3, 4, 5))
        self.assertTupleEqual((4, 5), actual)

    def test_set(self):
        actual = self.f({1, 2, 3, 4, 5})
        self.assertSetEqual({4, 5}, actual)

    def test_wrong_iterable_raises(self):
        with self.assertRaises(FilterError):
            _ = self.f(iter([1, 2, 3, 4, 5]))

    def test_wrong_criterion_raises(self):
        expected = ('Error calling\n'
                    'lambda\n'
                    'on element #2:\n'
                    '0\n'
                    'ZeroDivisionError:\n'
                    'division by zero')
        f = Filter(lambda x: 1 / x)
        with self.assertRaises(FilterError) as error:
            _ = f([1, 2, 0, 2, 1])
        self.assertEqual(expected, str(error.exception))


class TestWrapperAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Filter(greater_3, tuple)

    def test_has_criterion(self):
        f = Filter(greater_3, tuple)
        self.assertTrue(hasattr(f, 'criterion'))

    def test_criterion_is_none(self):
        f = Filter(greater_3, tuple)
        self.assertIs(f.criterion, greater_3)

    def test_has_wrapper(self):
        f = Filter(greater_3, tuple)
        self.assertTrue(hasattr(f, 'wrapper'))

    def test_wrapper_is_none(self):
        f = Filter(greater_3, tuple)
        self.assertIs(tuple, f.wrapper)

    def test_callable(self):
        f = Filter(greater_3, tuple)
        self.assertTrue(callable(f))


class TestWrapperUsage(unittest.TestCase):

    def test_wrapper_called_once(self):
        mock = Mock()
        f = Filter(greater_3, mock)
        _ = f([1, 2, 3, 4, 5])
        mock.assert_called_once()

    def test_wrapper_called_once_correctly(self):
        mock = Mock()
        f = Filter(greater_3, mock)
        _ = f([1, 2, 3, 4, 5])
        mock.assert_called_once_with([4, 5])

    def test_empty_list(self):
        f = Filter(greater_3, tuple)
        actual = f([])
        self.assertTupleEqual(tuple(), actual)

    def test_empty_tuple(self):
        f = Filter(greater_3, tuple)
        actual = f(tuple())
        self.assertTupleEqual(tuple(), actual)

    def test_empty_set(self):
        f = Filter(greater_3, tuple)
        actual = f(set())
        self.assertTupleEqual(tuple(), actual)

    def test_list(self):
        f = Filter(greater_3, tuple)
        actual = f([1, 2, 3, 4, 5])
        self.assertTupleEqual((4, 5), actual)

    def test_tuple(self):
        f = Filter(greater_3, tuple)
        actual = f((1, 2, 3, 4, 5))
        self.assertTupleEqual((4, 5), actual)

    def test_set(self):
        f = Filter(greater_3, tuple)
        actual = f({1, 2, 3, 4, 5})
        self.assertTupleEqual((4, 5), actual)

    def test_wrong_wrapper_raises(self):
        expected = 'Could not wrap filter results into an instance of int!'
        f = Filter(greater_3, int)
        with self.assertRaises(FilterError) as error:
            _ = f([1, 2, 3, 4, 5])
        self.assertEqual(expected, str(error.exception))


class TestMisc(unittest.TestCase):

    def test_annotate_wrapper(self):
        _ = Filter[int, list](greater_3)

    def test_annotate_wrapped_elements(self):
        _ = Filter[int, list[int]](greater_3)

    def test_default_pickle_works(self):
        f = Filter()
        _ = pickle.dumps(f)

    def test_default_repr(self):
        f = Filter()
        self.assertEqual('Filter(None, None)', repr(f))

    def test_criterion_pickle_works(self):
        f = Filter(greater_3)
        _ = pickle.dumps(f)

    def test_lambda_criterion_pickle_raises(self):
        f = Filter(lambda x: x > 3)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(f)

    def test_wrapper_pickle_works(self):
        f = Filter(greater_3, tuple)
        _ = pickle.dumps(f)

    def test_lambda_wrapper_pickle_raises(self):
        f = Filter(greater_3, lambda x: tuple(x))
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(f)

    def test_criterion_repr(self):
        f = Filter(greater_3)
        self.assertEqual('Filter(greater_3, None)', repr(f))

    def test_wrapper_repr(self):
        f = Filter(wrapper=tuple)
        self.assertEqual('Filter(None, tuple)', repr(f))

    def test_criterion_wrapper_repr(self):
        f = Filter(greater_3, tuple)
        self.assertEqual('Filter(greater_3, tuple)', repr(f))


if __name__ == '__main__':
    unittest.main()
