import unittest
from swak.jsonobject import JsonObject


class TestInheritance(unittest.TestCase):

    class A(JsonObject):
        a: int

    class Adef(JsonObject):
        a: int = 1

    def test_add_field(self):

        class AddB(self.A):
            b: str

        add_b = AddB(a=1, b='foo')
        self.assertTrue(hasattr(add_b, 'b'))
        self.assertIsInstance(add_b.b, str)
        self.assertEqual(add_b.b, 'foo')

    def test_add_new_default_field(self):

        class AddBdef(self.A):
            b: str = 'foo'

        add_b_def = AddBdef(a=1)
        self.assertTrue(hasattr(add_b_def, 'b'))
        self.assertIsInstance(add_b_def.b, str)
        self.assertEqual(add_b_def.b, 'foo')

    def test_add_default_to_existing_field(self):

        class AddAdef(self.A):
            a: int = 2

        add_a_def = AddAdef()
        self.assertTrue(hasattr(add_a_def, 'a'))
        self.assertIsInstance(add_a_def.a, int)
        self.assertEqual(add_a_def.a, 2)

    def test_change_default_of_existing_field(self):

        class ChangeAdef(self.Adef):
            a: int = 2

        change_a_def = ChangeAdef()
        self.assertTrue(hasattr(change_a_def, 'a'))
        self.assertIsInstance(change_a_def.a, int)
        self.assertEqual(change_a_def.a, 2)

    def test_cant_remove_default_of_existing_field(self):

        class AundefA(self.Adef):
            a: int

        a_undef_a = AundefA()
        self.assertTrue(hasattr(a_undef_a, 'a'))
        self.assertIsInstance(a_undef_a.a, int)
        self.assertEqual(a_undef_a.a, 1)

    def test_change_type_of_existing_field(self):

        class AchangeA(self.A):
            a: str

        a_change_a = AchangeA(a=1)
        self.assertTrue(hasattr(a_change_a, 'a'))
        self.assertIsInstance(a_change_a.a, str)
        self.assertEqual(a_change_a.a, '1')

    def test_change_type_of_existing_field_and_set_default(self):

        class AchangeAdef(self.A):
            a: str = 'foo'

        a_change_a_def = AchangeAdef()
        self.assertTrue(hasattr(a_change_a_def, 'a'))
        self.assertIsInstance(a_change_a_def.a, str)
        self.assertEqual(a_change_a_def.a, 'foo')

    def test_change_type_of_existing_field_and_change_default(self):

        class AchangeAdef(self.Adef):
            a: str = 'foo'

        a_change_a_def = AchangeAdef()
        self.assertTrue(hasattr(a_change_a_def, 'a'))
        self.assertIsInstance(a_change_a_def.a, str)
        self.assertEqual(a_change_a_def.a, 'foo')

    def test_grandparent(self):

        class B(self.A):
            b: str = 'foo'

        class C(B):
            c: float = 2.0

        c = C(a=2)
        self.assertTrue(hasattr(c, 'a'))
        self.assertIsInstance(c.a, int)
        self.assertEqual(c.a, 2)

    def test_multiple_inheritance(self):

        class B(JsonObject):
            b: str = 'foo'

        class C(self.A, B):
            c: float = 2.0

        c = C(a=2)
        self.assertTrue(hasattr(c, 'a'))
        self.assertIsInstance(c.a, int)
        self.assertEqual(c.a, 2)
        self.assertTrue(hasattr(c, 'b'))
        self.assertIsInstance(c.b, str)
        self.assertEqual(c.b, 'foo')

    def test_mro(self):

        class B(JsonObject):
            a: str

        class C(self.A, B):
            c: float

        class D(B, self.A):
            c: float

        c = C(a=2, c=3.0)
        self.assertTrue(hasattr(c, 'a'))
        self.assertIsInstance(c.a, int)
        self.assertEqual(c.a, 2)

        d = D(a=2, c=3.0)
        self.assertTrue(hasattr(d, 'a'))
        self.assertIsInstance(d.a, str)
        self.assertEqual(d.a, '2')

    def test_class_kwargs_added(self):

        class B(self.A, ignore_extra='foo', raise_extra='bar'):
            pass

        b = B(a=1)
        self.assertTrue(hasattr(b, 'ignore_extra'))
        self.assertIsInstance(b.ignore_extra, str)
        self.assertEqual('foo', b.ignore_extra)
        self.assertTrue(hasattr(b, 'raise_extra'))
        self.assertIsInstance(b.raise_extra, str)
        self.assertEqual('bar', b.raise_extra)

    def test_class_kwargs_not_inherited(self):

        class A(
                JsonObject,
                ignore_extra='foo',
                raise_extra='bar'
        ):
            a: int

        class B(A):
            pass

        b = B(a=1)
        self.assertTrue(hasattr(b, 'ignore_extra'))
        self.assertIsInstance(b.ignore_extra, bool)
        self.assertTrue(hasattr(b, 'raise_extra'))
        self.assertIsInstance(b.raise_extra, bool)


if __name__ == '__main__':
    unittest.main()
