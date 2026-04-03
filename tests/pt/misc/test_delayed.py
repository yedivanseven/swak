import pickle
import unittest
from unittest.mock import Mock
from swak.pt.misc import Delayed


def f(*_, **__):
    pass


class G:

    def __init__(self, *_, **__):
        pass

    def h(self, *_, **__):
        pass


class TestArguments(unittest.TestCase):

    def setUp(self):
        self.args = 1, 2
        self.kwargs = {'three': 3, 'four': 4}
        self.d = Delayed(f, *self.args, **self.kwargs)

    def test_has_call(self):
        self.assertTrue(hasattr(self.d, 'call'))

    def test_call(self):
        self.assertIs(self.d.call, f)

    def test_has_args(self):
        self.assertTrue(hasattr(self.d, 'args'))

    def test_args(self):
        self.assertTupleEqual(self.args, self.d.args)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.d, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.d.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.args = 1, 2
        self.kwargs = {'three': 3, 'four': 4}

    def test_non_recursive_no_args(self):
        expected = 'mock return'
        call = Mock(return_value=expected)

        args = 1, 2
        kwargs = {'three': 3, 'four': 4}

        delayed = Delayed(call, *args, **kwargs)
        call.assert_not_called()
        actual = delayed()
        call.assert_called_once_with(*args, **kwargs)
        self.assertEqual(expected, actual)

    def test_non_recursive_args(self):
        expected = 'mock return'
        call = Mock(return_value=expected)

        args = 1, 2
        kwargs = {'three': 3, 'four': 4}

        delayed = Delayed(call, *args, **kwargs)
        call.assert_not_called()
        actual = delayed('these', are='disregarded')
        call.assert_called_once_with(*args, **kwargs)
        self.assertEqual(expected, actual)

    def test_recursive_nor_args(self):
        root = Mock(return_value='root')
        child = Mock(return_value='child')
        grandchild = Mock(return_value='grandchild')

        delayed_grandchild = Delayed(grandchild, 1, answer=42)
        grandchild.assert_not_called()
        delayed_child = Delayed(child, delayed_grandchild, hello='world')
        grandchild.assert_not_called()
        delayed_root = Delayed(root, 2, child=delayed_child)
        root.assert_not_called()

        actual = delayed_root()
        self.assertEqual('root', actual)

        grandchild.assert_called_once_with(1, answer=42)
        child.assert_called_once_with('grandchild', hello='world')
        root.assert_called_once_with(2, child='child')

    def test_recursive_args(self):
        root = Mock(return_value='root')
        child = Mock(return_value='child')
        grandchild = Mock(return_value='grandchild')

        delayed_grandchild = Delayed(grandchild, 1, answer=42)
        grandchild.assert_not_called()
        delayed_child = Delayed(child, delayed_grandchild, hello='world')
        grandchild.assert_not_called()
        delayed_root = Delayed(root, 2, child=delayed_child)
        root.assert_not_called()

        actual = delayed_root('these', are='disregarded')
        self.assertEqual('root', actual)

        grandchild.assert_called_once_with(1, answer=42)
        child.assert_called_once_with('grandchild', hello='world')
        root.assert_called_once_with(2, child='child')


class TestMisc(unittest.TestCase):

    def test_repr_simple_function(self):
        delayed_f = Delayed(f)
        expected = 'Delayed(f)'
        self.assertEqual(expected, repr(delayed_f))

    def test_repr_simple_class(self):
        delayed_f = Delayed(G)
        expected = 'Delayed(G)'
        self.assertEqual(expected, repr(delayed_f))

    def test_repr_simple_method(self):
        g = G()
        delayed_f = Delayed(g.h)
        expected = 'Delayed(G.h)'
        self.assertEqual(expected, repr(delayed_f))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(Delayed(G().h, Delayed(f))))

    def test_pickle_raises_with_lambda(self):
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(
                Delayed(G().h, Delayed(lambda x: x))
            ))


if __name__ == '__main__':
    unittest.main()
