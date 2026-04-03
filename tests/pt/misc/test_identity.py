import unittest
from swak.pt.misc import identity, Identity


class TestIdentityFunction(unittest.TestCase):

    def test_callable(self):
        self.assertTrue(callable(identity))

    def test_arg(self):
        obj = object()
        result = identity(obj)
        self.assertIs(result, obj)

    def test_arg_and_args(self):
        obj = object()
        result = identity(obj, 'bar', answer=42)
        self.assertIs(result, obj)

    def test_arg_and_kwargs(self):
        obj = object()
        result = identity(obj, foo='bar', answer=42)
        self.assertIs(result, obj)

    def test_arg_and_args_and_kwargs(self):
        obj = object()
        result = identity(obj, 'bar', answer=42)
        self.assertIs(result, obj)


class TestIdentity(unittest.TestCase):

    def test_empty_instantiation(self):
        _ = Identity()

    def test_instantiation_accepts_args(self):
        _ = Identity(1, 'foo', 4.2)

    def test_instantiation_accepts_kwargs(self):
        _ = Identity(foo=1, bar='baz', answer=4.2)

    def test_instantiation_accepts_args_and_kwargs(self):
        _ = Identity(1, 42, foo='bar')

    def test_callable(self):
        self.assertTrue(callable(Identity()))

    def test_arg(self):
        obj = object()
        result = Identity()(obj)
        self.assertIs(result, obj)

    def test_arg_and_args(self):
        obj = object()
        result = Identity()(obj, 'bar', 42)
        self.assertIs(result, obj)

    def test_arg_and_kwargs(self):
        obj = object()
        result = Identity()(obj, foo='bar', answer=42)
        self.assertIs(result, obj)

    def test_arg_and_args_and_kwargs(self):
        obj = object()
        result = Identity()(obj, 'bar', answer=42)
        self.assertIs(result, obj)

    def test_has_reset_parameters(self):
        i = Identity()
        self.assertTrue(hasattr(i, 'reset_parameters'))

    def test_reset_parameters(self):
        i = Identity()
        self.assertTrue(callable(i.reset_parameters))

    def test_call_reset_parameters(self):
        i = Identity()
        i.reset_parameters()


if __name__ == '__main__':
    unittest.main()
