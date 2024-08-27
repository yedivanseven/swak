import unittest
from swak.pt.misc import identity, Identity


class TestIdentityFunction(unittest.TestCase):

    def test_callable(self):
        self.assertTrue(callable(Identity()))

    def test_empty(self):
        result = identity()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_empty_and_kwargs(self):
        result = identity(foo='bar', answer=42)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_one_arg(self):
        obj = object()
        result = identity(obj)
        self.assertIs(result, obj)

    def test_one_arg_and_kwargs(self):
        obj = object()
        result = identity(obj, foo='bar', answer=42)
        self.assertIs(result, obj)

    def test_args(self):
        args = 1, 'foo'
        result = identity(*args)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual(args, result)

    def test_args_and_kwargs(self):
        args = 1, 'foo'
        result = identity(*args, foo='bar', answer=42)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual(args, result)


class TestIdentityModule(unittest.TestCase):

    def test_empty_instantiation(self):
        _ = Identity()

    def test_instantiation_accepts_args(self):
        _ = Identity(1, 'foo', 4.2)

    def test_instantiation_accepts_kwargs(self):
        _ = Identity(foo=1, bar='baz', answer=4.2)

    def test_instantiation_accepts_kargs_and_kwargs(self):
        _ = Identity(1, 42, foo='bar')

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
