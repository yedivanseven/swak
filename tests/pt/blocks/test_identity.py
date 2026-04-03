import unittest
from swak.pt.blocks import IdentityBlock


class TestIdentityBlock(unittest.TestCase):

    def test_empty_instantiation(self):
        _ = IdentityBlock(4)

    def test_instantiation_accepts_args(self):
        _ = IdentityBlock(4, 'foo', 4.2)

    def test_instantiation_accepts_kwargs(self):
        _ = IdentityBlock(4, foo=1, bar='baz', answer=4.2)

    def test_instantiation_accepts_args_and_kwargs(self):
        _ = IdentityBlock(4, 42, foo='bar')

    def test_callable(self):
        self.assertTrue(callable(IdentityBlock(4)))

    def test_arg(self):
        obj = object()
        result = IdentityBlock(4)(obj)
        self.assertIs(result, obj)

    def test_arg_and_args(self):
        obj = object()
        result = IdentityBlock(4)(obj, 'bar', 42)
        self.assertIs(result, obj)

    def test_arg_and_kwargs(self):
        obj = object()
        result = IdentityBlock(4)(obj, foo='bar', answer=42)
        self.assertIs(result, obj)

    def test_arg_and_args_and_kwargs(self):
        obj = object()
        result = IdentityBlock(4)(obj, 'bar', answer=42)
        self.assertIs(result, obj)

    def test_has_reset_parameters(self):
        i = IdentityBlock(4)
        self.assertTrue(hasattr(i, 'reset_parameters'))

    def test_reset_parameters(self):
        i = IdentityBlock(4)
        self.assertTrue(callable(i.reset_parameters))

    def test_call_reset_parameters(self):
        i = IdentityBlock(4)
        i.reset_parameters()

    def test_has_mod_dim(self):
        i = IdentityBlock(4)
        self.assertTrue(hasattr(i, 'mod_dim'))

    def test_mod_dim(self):
        i = IdentityBlock(4)
        self.assertIsInstance(i.mod_dim, int)
        self.assertEqual(4, i.mod_dim)

    def test_has_device(self):
        i = IdentityBlock(4)
        self.assertTrue(hasattr(i, 'device'))

    def test_device(self):
        i = IdentityBlock(4)
        self.assertIsNone(i.device)

    def test_has_dtype(self):
        i = IdentityBlock(4)
        self.assertTrue(hasattr(i, 'dtype'))

    def test_dtype(self):
        i = IdentityBlock(4)
        self.assertIsNone(i.dtype)

    def test_has_new(self):
        i = IdentityBlock(4)
        self.assertTrue(hasattr(i, 'new'))

    def test_new(self):
        i = IdentityBlock(4)
        self.assertTrue(callable(i.new))

    def test_call_new(self):
        old = IdentityBlock(4)
        new = old.new()
        self.assertIsInstance(new, IdentityBlock)
        self.assertIsNot(old, new)


if __name__ == '__main__':
    unittest.main()
