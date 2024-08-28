import unittest
from swak.pt.misc import identity, Identity


class TestIdentityFunction(unittest.TestCase):

    def test_callable(self):
        self.assertTrue(callable(Identity()))

    def test_arg(self):
        obj = object()
        result = identity(obj)
        self.assertIs(result, obj)

    def test_arg_and_kwargs(self):
        obj = object()
        result = identity(obj, foo='bar', answer=42)
        self.assertIs(result, obj)


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

    def test_arg(self):
        obj = object()
        result = Identity()(obj)
        self.assertIs(result, obj)

    def test_arg_and_kwargs(self):
        obj = object()
        result = Identity()(obj, foo='bar', answer=42)
        self.assertIs(result, obj)


if __name__ == '__main__':
    unittest.main()
