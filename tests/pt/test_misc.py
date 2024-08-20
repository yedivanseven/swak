import unittest
from swak.pt.misc import Identity


class TestIdentity(unittest.TestCase):

    def test_instantiation(self):
        _ = Identity()

    def test_callable(self):
        self.assertTrue(callable(Identity()))

    def test_empty(self):
        result = Identity()()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_empty_and_kwargs(self):
        result = Identity()(foo='bar', answer=42)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_one_arg(self):
        obj = object()
        result = Identity()(obj)
        self.assertIs(result, obj)

    def test_one_arg_and_kwargs(self):
        obj = object()
        result = Identity()(obj, foo='bar', answer=42)
        self.assertIs(result, obj)

    def test_args(self):
        args = 1, 'foo'
        result = Identity()(*args)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual(args, result)

    def test_args_and_kwargs(self):
        args = 1, 'foo'
        result = Identity()(*args, foo='bar', answer=42)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual(args, result)


if __name__ == '__main__':
    unittest.main()
