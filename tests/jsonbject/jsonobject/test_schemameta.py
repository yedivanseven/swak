import unittest
from unittest.mock import Mock
from swak.jsonobject.jsonobject import SchemaMeta
from swak.jsonobject.fields import Maybe
from swak.jsonobject.exceptions import ValidationErrors


class TestEmptyAttributes(unittest.TestCase):

    def test_instantiation(self):

        class Empty(metaclass=SchemaMeta):
            pass

    def test_has_schema(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertTrue(hasattr(Empty, '__annotations__'))

    def test_schema(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertDictEqual({}, Empty.__annotations__)

    def test_has_defaults(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertTrue(hasattr(Empty, '__defaults__'))

    def test_defaults(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertDictEqual({}, Empty.__defaults__)


class TestAttributes(unittest.TestCase):

    def test_has_schema(self):

        class Test(metaclass=SchemaMeta):
            a: int
            b: str

        self.assertTrue(hasattr(Test, '__annotations__'))

    def test_schema(self):

        class Test(metaclass=SchemaMeta):
            a: int
            b: str

        self.assertDictEqual({'a': int, 'b': str}, Test.__annotations__)

    def test_has_mixed_defaults(self):

        class Test(metaclass=SchemaMeta):
            a: int
            b: str = 'hello'

        self.assertTrue(hasattr(Test, '__annotations__'))
        self.assertDictEqual({'a': int, 'b': str}, Test.__annotations__)
        self.assertTrue(hasattr(Test, '__defaults__'))
        self.assertDictEqual({'b': 'hello'}, Test.__defaults__)

    def test_mixed_defaults(self):

        class Test(metaclass=SchemaMeta):
            a: int
            b: str = 'hello'

        self.assertTrue(hasattr(Test, '__annotations__'))
        self.assertDictEqual({'a': int, 'b': str}, Test.__annotations__)
        self.assertDictEqual({'b': 'hello'}, Test.__defaults__)

    def test_has_full_defaults(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1
            b: str = 'hello'

        self.assertTrue(hasattr(Test, '__annotations__'))
        self.assertDictEqual({'a': int, 'b': str}, Test.__annotations__)
        self.assertTrue(hasattr(Test, '__defaults__'))

    def test_full_defaults(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1
            b: str = 'hello'

        self.assertTrue(hasattr(Test, '__annotations__'))
        self.assertDictEqual({'a': int, 'b': str}, Test.__annotations__)
        self.assertDictEqual({'a': 1, 'b': 'hello'}, Test.__defaults__)

    def test_caster_called(self):

        mock = Mock()

        class Test(metaclass=SchemaMeta):
            a: int = 1.0
            b: mock = '2.0'

        mock.assert_called_once()
        mock.assert_called_once_with('2.0')

    def test_cast_defaults_type(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1.0
            b: float = '2.0'

        self.assertIsInstance(Test.__defaults__['b'], float)

    def test_cast_defaults_value(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1.0
            b: float = '2.0'

        self.assertEqual(2.0, Test.__defaults__['b'])

    def test_has_class_variable(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1
            b: str = 'hello'
            c = 2.0

        self.assertTrue(hasattr(Test, 'c'))

    def test_class_variable_type(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1
            b: str = 'hello'
            c = 2.0

        self.assertIsInstance(Test.c, float)

    def test_class_variable_value(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1
            b: str = 'hello'
            c = 2.0

        self.assertEqual(2.0, Test.c)


class TestValidations(unittest.TestCase):

    def test_annotation_not_callable(self):

        with self.assertRaises(ValidationErrors):

            class Test(metaclass=SchemaMeta):
                a: int
                b: 1

    def test_field_blacklisted(self):

        with self.assertRaises(ValidationErrors):

            class Test(metaclass=SchemaMeta):
                a: int
                keys: str

    def test_field_double_underscore(self):

        with self.assertRaises(ValidationErrors):

            class Test(metaclass=SchemaMeta):
                a: int
                __b: str

    def test_combi_message(self):

        with self.assertRaises(ValidationErrors):

            class Test(metaclass=SchemaMeta):
                a: int
                b: 1
                __c: str

    def test_default_none(self):

        with self.assertRaises(ValidationErrors):

            class Test(metaclass=SchemaMeta):
                a: int
                b: str = None

    def test_default_wrong_type(self):

        with self.assertRaises(ValidationErrors):

            class Test(metaclass=SchemaMeta):
                a: str
                b: int = '1.0'


class TestMaybe(unittest.TestCase):

    def test_maybe_works_with_none(self):

        class Test(metaclass=SchemaMeta):
            a: Maybe[int](int) = None
            b: str = 'foo'
            c: float

        self.assertIsNone(Test.__defaults__['a'])

    def test_maybe_works_with_value(self):

        class Test(metaclass=SchemaMeta):
            a: Maybe[int](int) = 1.0
            b: str = 'foo'
            c: float

        self.assertIsInstance(Test.__defaults__['a'], int)
        self.assertEqual(1, Test.__defaults__['a'])

    def test_maybe_works_with_null_str(self):

        class Test(metaclass=SchemaMeta):
            a: Maybe[int](int) = 'null'
            b: str = 'foo'
            c: float

        self.assertIsNone(Test.__defaults__['a'])

    def test_maybe_works_with_none_str(self):

        class Test(metaclass=SchemaMeta):
            a: Maybe[int](int) = 'None'
            b: str = 'foo'
            c: float

        self.assertIsNone(Test.__defaults__['a'])

    def test_maybe_works_without_default(self):

        class Test(metaclass=SchemaMeta):
            a: Maybe[int](int)
            b: str = 'foo'
            c: float

        self.assertDictEqual({'b': 'foo'}, Test.__defaults__)

    def test_maybe_lets_cast_error_through(self):

        with self.assertRaises(ValidationErrors):

            class Test(metaclass=SchemaMeta):
                a: Maybe[int](int) = '1.0'
                b: str = 'foo'
                c: float


class TestExtraNone(unittest.TestCase):

    def test_has_ignore_extra(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertTrue(hasattr(Empty, '__ignore_extra__'))

    def test_ignore_extra(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertFalse(Empty.__ignore_extra__)

    def test_has_raise_extra(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertTrue(hasattr(Empty, '__raise_extra__'))

    def test_raise_extra(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertTrue(Empty.__raise_extra__)

    def test_has_respect_none(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertTrue(hasattr(Empty, '__respect_none__'))

    def test_respect_none(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertFalse(Empty.__respect_none__)


class TestKwargFields(unittest.TestCase):

    def test_add_kwarg_field_string(self):

        class Test(metaclass=SchemaMeta, foo='bar'):
            a: int = 1
            b: float

        expected_schema = {'a': int, 'b': float, 'foo': str}
        expected_defaults = {'a': 1, 'foo': 'foo'}
        self.assertDictEqual(expected_schema, Test.__annotations__)
        self.assertDictEqual(expected_defaults, Test.__defaults__)

    def test_add_kwarg_field_float(self):

        class Test(metaclass=SchemaMeta, foo=1.0):
            a: int = 1
            b: float

        expected_schema = {'a': int, 'b': float, 'foo': str}
        expected_defaults = {'a': 1, 'foo': 'foo'}
        self.assertDictEqual(expected_schema, Test.__annotations__)
        self.assertDictEqual(expected_defaults, Test.__defaults__)

    def test_body_field_overwrites_kwarg_field_and_default(self):

        class Test(metaclass=SchemaMeta, b=True):
            a: int = 1
            b: float

        expected_schema = {'a': int, 'b': float}
        expected_defaults = {'a': 1}
        self.assertDictEqual(expected_schema, Test.__annotations__)
        self.assertDictEqual(expected_defaults, Test.__defaults__)

    def test_body_default_overwrites_kwarg_default(self):

        class Test(metaclass=SchemaMeta, a=True):
            a: int = 1
            b: float

        expected_schema = {'a': int, 'b': float}
        expected_defaults = {'a': 1}
        self.assertDictEqual(expected_schema, Test.__annotations__)
        self.assertDictEqual(expected_defaults, Test.__defaults__)


class TestInheritance(unittest.TestCase):

    class A(metaclass=SchemaMeta):
        a: int

    class Adef(metaclass=SchemaMeta):
        a: int = 1

    def test_add_field(self):

        class AddB(self.A):
            b: str

        self.assertIsInstance(AddB, SchemaMeta)
        self.assertDictEqual({'a': int, 'b': str}, AddB.__annotations__)
        self.assertDictEqual({}, AddB.__defaults__)

    def test_add_new_default_field(self):

        class AddBdef(self.A):
            b: str = 'hello'

        self.assertIsInstance(AddBdef, SchemaMeta)
        self.assertDictEqual({'a': int, 'b': str}, AddBdef.__annotations__)
        self.assertDictEqual({'b': 'hello'}, AddBdef.__defaults__)

    def test_add_default_to_existing_field(self):

        class AddAdef(self.A):
            a: int = 2

        self.assertIsInstance(AddAdef, SchemaMeta)
        self.assertDictEqual({'a': int}, AddAdef.__annotations__)
        self.assertDictEqual({'a': 2}, AddAdef.__defaults__)

    def test_change_default_of_existing_field(self):

        class ChangeAdef(self.Adef):
            a: int = 2

        self.assertIsInstance(ChangeAdef, SchemaMeta)
        self.assertDictEqual({'a': int}, ChangeAdef.__annotations__)
        self.assertDictEqual({'a': 2}, ChangeAdef.__defaults__)

    def test_cant_remove_default_of_existing_field(self):

        class AundefA(self.Adef):
            a: int

        self.assertIsInstance(AundefA, SchemaMeta)
        self.assertDictEqual({'a': int}, AundefA.__annotations__)
        self.assertDictEqual({'a': 1}, AundefA.__defaults__)

    def test_change_type_of_existing_field(self):

        class AchangeA(self.A):
            a: str

        self.assertIsInstance(AchangeA, SchemaMeta)
        self.assertDictEqual({'a': str}, AchangeA.__annotations__)
        self.assertDictEqual({}, AchangeA.__defaults__)

    def test_change_type_of_existing_field_and_set_default(self):

        class AchangeAdef(self.A):
            a: str = 'foo'

        self.assertIsInstance(AchangeAdef, SchemaMeta)
        self.assertDictEqual({'a': str}, AchangeAdef.__annotations__)
        self.assertDictEqual({'a': 'foo'}, AchangeAdef.__defaults__)

    def test_change_type_of_existing_field_and_change_default(self):

        class AchangeAdef(self.Adef):
            a: str = 'foo'

        self.assertIsInstance(AchangeAdef, SchemaMeta)
        self.assertDictEqual({'a': str}, AchangeAdef.__annotations__)
        self.assertDictEqual({'a': 'foo'}, AchangeAdef.__defaults__)

    def test_grandparent(self):

        class B(self.A):
            b: str = 'foo'

        class C(B):
            c: float = 2.0

        self.assertIsInstance(C, SchemaMeta)
        self.assertDictEqual(
            {'a': int, 'b': str, 'c': float},
            C.__annotations__
        )
        self.assertDictEqual({'b': 'foo', 'c': 2.0}, C.__defaults__)

    def test_multiple_inheritance(self):

        class B(metaclass=SchemaMeta):
            b: str = 'foo'

        class C(self.A, B):
            c: float = 2.0

        self.assertIsInstance(C, SchemaMeta)
        self.assertDictEqual(
            {'a': int, 'b': str, 'c': float},
            C.__annotations__
        )
        self.assertDictEqual({'b': 'foo', 'c': 2.0}, C.__defaults__)

    def test_mro(self):

        class B(metaclass=SchemaMeta):
            a: str

        class C(self.A, B):
            c: float

        class D(B, self.A):
            c: float

        self.assertIsInstance(C, SchemaMeta)
        self.assertDictEqual({'a': int, 'c': float}, C.__annotations__)
        self.assertDictEqual({}, C.__defaults__)
        self.assertDictEqual({'a': str, 'c': float}, D.__annotations__)
        self.assertDictEqual({}, D.__defaults__)

    def test_has_ignore_extra(self):

        class IgnoreExtra(self.A, ignore_extra=True):
            pass

        self.assertTrue(hasattr(IgnoreExtra, '__ignore_extra__'))

    def test_ignore_extra(self):

        class IgnoreExtra(self.A, ignore_extra=True):
            pass

        self.assertTrue(IgnoreExtra.__ignore_extra__)

    def test_ignore_extra_inherited(self):

        class IgnoreExtra(self.A, ignore_extra=True):
            pass

        class Child(IgnoreExtra):
            pass

        self.assertTrue(hasattr(Child, '__ignore_extra__'))
        self.assertTrue(Child.__ignore_extra__)

    def test_has_raise_extra(self):

        class RaiseExtra(self.A, raise_extra=False):
            pass

        self.assertTrue(hasattr(RaiseExtra, '__raise_extra__'))

    def test_raise_extra(self):

        class RaiseExtra(self.A, raise_extra=False):
            pass

        self.assertFalse(RaiseExtra.__raise_extra__)

    def test_raise_extra_inherited(self):

        class RaiseExtra(self.A, raise_extra=False):
            pass

        class Child(RaiseExtra):
            pass

        self.assertTrue(hasattr(Child, '__raise_extra__'))
        self.assertFalse(Child.__raise_extra__)

    def test_has_respect_none(self):

        class RespectNone(self.A, respect_none=True):
            pass

        self.assertTrue(hasattr(RespectNone, '__respect_none__'))

    def test_respect_none(self):

        class RespectNone(self.A, respect_none=True):
            pass

        self.assertTrue(RespectNone.__respect_none__)

    def test_respect_none_inherited(self):

        class RespectNone(self.A, respect_none=True):
            pass

        class Child(RespectNone):
            pass

        self.assertTrue(hasattr(Child, '__respect_none__'))
        self.assertTrue(Child.__respect_none__)

    def test_add_kwarg_field_string(self):

        class Test(self.A, foo='bar'):
            pass

        expected_schema = {'a': int, 'foo': str}
        expected_defaults = {'foo': 'foo'}
        self.assertDictEqual(expected_schema, Test.__annotations__)
        self.assertDictEqual(expected_defaults, Test.__defaults__)

    def test_add_kwarg_field_float(self):

        class Test(self.A, foo=1.0):
            pass

        expected_schema = {'a': int, 'foo': str}
        expected_defaults = {'foo': 'foo'}
        self.assertDictEqual(expected_schema, Test.__annotations__)
        self.assertDictEqual(expected_defaults, Test.__defaults__)

    def test_add_kwarg_field_inherited(self):

        class Test(self.A, foo='bar'):
            pass

        class Child(Test):
            pass

        expected_schema = {'a': int, 'foo': str}
        expected_defaults = {'foo': 'foo'}
        self.assertDictEqual(expected_schema, Child.__annotations__)
        self.assertDictEqual(expected_defaults, Child.__defaults__)

    def test_body_field_overwrites_kwarg_field_and_default(self):

        class Test(self.A, a=True):
            pass

        expected_schema = {'a': int}
        expected_defaults = {}
        self.assertDictEqual(expected_schema, Test.__annotations__)
        self.assertDictEqual(expected_defaults, Test.__defaults__)

    def test_body_default_overwrites_kwarg_default(self):

        class Test(self.Adef, a=True):
            a: int = 1

        expected_schema = {'a': int}
        expected_defaults = {'a': 1}
        self.assertDictEqual(expected_schema, Test.__annotations__)
        self.assertDictEqual(expected_defaults, Test.__defaults__)


if __name__ == '__main__':
    unittest.main()
