import pickle
import unittest
from unittest.mock import Mock
from swak.pd import GroupByApply


def f(x):
    return x.mean()


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.apply = GroupByApply(f)

    def test_has_func(self):
        self.assertTrue(hasattr(self.apply, 'func'))

    def test_func(self):
        self.assertIs(self.apply.func, f)

    def test_has_args(self):
        self.assertTrue(hasattr(self.apply, 'args'))

    def test_args(self):
        self.assertTupleEqual((), self.apply.args)

    def test_has_kwargs(self):
        self.assertTrue(self.apply, 'kwargs')

    def test_kwargs(self):
        self.assertDictEqual({}, self.apply.kwargs)



class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.args = 'foo', 1
        self.kwargs = {'bar': 2, 'answer': 42}
        self.apply = GroupByApply(f,*self.args, **self.kwargs)

    def test_args(self):
        self.assertTupleEqual(self.args, self.apply.args)

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.apply.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.args = 'foo', 1
        self.kwargs = {'bar': 2, 'answer': 42}
        self.apply = GroupByApply(f,*self.args, **self.kwargs)

    def test_callable(self):
            self.assertTrue(callable(self.apply))

    def test_apply_called(self):
        grouped = Mock()
        _ = self.apply(grouped)
        grouped.apply.assert_called_once_with(f, *self.args, **self.kwargs)

    def test_return_value(self):
        grouped = Mock()
        grouped.apply = Mock(return_value='cheese')
        actual = self.apply(grouped)
        self.assertEqual('cheese', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        apply = GroupByApply(f)
        expected = 'GroupByApply(f)'
        self.assertEqual(expected, repr(apply))

    def test_custom_repr(self):
        apply = GroupByApply(f, 'foo', 1, answer=42)
        expected = "GroupByApply(f, 'foo', 1, answer=42)"
        self.assertEqual(expected, repr(apply))

    def test_pickle_works_with_function(self):
        apply = GroupByApply(f)
        _ = pickle.loads(pickle.dumps(apply))

    def test_pickle_raises_with_lambda(self):
        apply = GroupByApply(lambda x: x.mean())
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(apply))


if __name__ == '__main__':
    unittest.main()
