import unittest
import pickle
from swak.magic import IndentRepr, ArgRepr


def f():
    pass


class Cls:

    @classmethod
    def c(cls):
        pass

    def m(self):
        pass

    @staticmethod
    def s():
        pass


class Call:

    def __call__(self):
        pass


class TestArgumentTypes(unittest.TestCase):

    class Empty(IndentRepr):

        def __init__(self) -> None:
            super().__init__()

    class Args(IndentRepr):

        def __init__(self, *args) -> None:
            super().__init__(*args)

    def test_empty_super_init(self):
        empty = self.Empty()
        self.assertEqual('Empty:\n', repr(empty))

    def test_no_args(self):
        no_args = self.Args()
        self.assertEqual('Args:\n', repr(no_args))

    def test_int_args(self):
        int_args = self.Args(1, 2, 3)
        expected = "Args:\n[ 0] 1\n[ 1] 2\n[ 2] 3"
        self.assertEqual(expected, repr(int_args))

    def test_callable_args(self):
        cls = Cls()
        string_args = self.Args(
            f,
            Cls,
            lambda x: x,
            Call(),
            cls.m,
            cls.c,
            cls.s
        )
        expected = ("Args:\n[ 0] f\n[ 1] Cls\n[ 2] lambda\n[ 3] Call(...)\n"
                    "[ 4] Cls.m\n[ 5] Cls.c\n[ 6] Cls.s")
        self.assertEqual(expected, repr(string_args))

    def test_custom_object_args(self):
        cls = Cls()
        string_args = self.Args(cls)
        expected = f"Args:\n[ 0] {str(cls)}"
        self.assertEqual(expected, repr(string_args))

    def test_string_args(self):
        string_args = self.Args('a', 'b', 'c')
        expected = "Args:\n[ 0] 'a'\n[ 1] 'b'\n[ 2] 'c'"
        self.assertEqual(expected, repr(string_args))

    def test_nested_once(self):
        child = self.Args(2, 3)
        parent = self.Args(1, child, 4)
        expected_child = "[ 1] Args:\n     [ 0] 2\n     [ 1] 3\n"
        expected = f"Args:\n[ 0] 1\n{expected_child}[ 2] 4"
        self.assertEqual(expected, repr(parent))

    def test_nested_twice(self):
        child = self.Args(3, 4)
        parent = self.Args(2, child, 5)
        grand = self.Args(1, parent, 6)
        ch = "     [ 1] Args:\n          [ 0] 3\n          [ 1] 4\n"
        pa = f"[ 1] Args:\n     [ 0] 2\n{ch}     [ 2] 5\n"
        expected = f"Args:\n[ 0] 1\n{pa}[ 2] 6"
        self.assertEqual(expected, repr(grand))


class TestInheritance(unittest.TestCase):

    def test_parent_indentrepr(self):

        class Parent(IndentRepr):

            def __init__(self, a):
                super().__init__(a)
                self.a = a

        class Child(Parent):

            def __init__(self, a):
                super().__init__(a)

        child = Child(1)
        self.assertEqual('Child:\n[ 0] 1', repr(child))
        self.assertTrue(hasattr(child, 'a'))
        self.assertEqual(1, child.a)

    def test_child_indentrepr_last(self):

        class Parent:

            def __init__(self, a):
                self.a = a

        class Child(Parent, IndentRepr):

            def __init__(self, a):
                super(Parent, self).__init__(a)
                super().__init__(a)

        child = Child(1)
        self.assertEqual('Child:\n[ 0] 1', repr(child))
        self.assertTrue(hasattr(child, 'a'))
        self.assertEqual(1, child.a)

    def test_child_indentrepr_first(self):

        class Parent:

            def __init__(self, a):
                self.a = a

        class Child(IndentRepr, Parent):

            def __init__(self, a):
                super(IndentRepr, self).__init__(a)
                super().__init__(a)

        child = Child(1)
        self.assertEqual('Child:\n[ 0] 1', repr(child))
        self.assertTrue(hasattr(child, 'a'))
        self.assertEqual(1, child.a)


class TestName(unittest.TestCase):

    class Indent(IndentRepr):
        pass

    def setUp(self) -> None:
        self.i = self.Indent()

    def test_method_present(self):
        self.assertTrue(hasattr(self.i, '_name'))

    def test_method_callable(self):
        self.assertTrue(callable(self.i._name))

    def test_return_type(self):
        actual = self.i._name(1)
        self.assertIsInstance(actual, str)

    def test_name_of_lambda(self):
        actual = self.i._name(lambda x: x)
        self.assertEqual('lambda', actual)

    def test_name_of_function(self):
        actual = self.i._name(f)
        self.assertEqual('f', actual)

    def test_name_of_class(self):
        actual = self.i._name(Cls)
        self.assertEqual('Cls', actual)

    def test_name_of_object(self):
        cls = Cls()
        actual = self.i._name(cls)
        self.assertEqual('Cls(...)', actual)

    def test_name_of_method(self):
        cls = Cls()
        actual = self.i._name(cls.m)
        self.assertEqual('Cls.m', actual)

    def test_name_of_staticmethod(self):
        cls = Cls()
        actual = self.i._name(cls.s)
        self.assertEqual('Cls.s', actual)

    def test_name_of_classmethod(self):
        actual = self.i._name(Cls.c)
        self.assertEqual('Cls.c', actual)

    def test_name_of_None(self):
        actual = self.i._name(None)
        self.assertEqual('None', actual)

    def test_argrepr(self):

        class Arg(ArgRepr):

            def __init__(self, b, c):
                super().__init__(b, c)

        arg = Arg('foo', 42)
        expected = "Arg('foo', 42)"
        actual = self.i._name(arg)
        self.assertEqual(expected, actual)

    def test_indentrepr(self):

        class Ind(IndentRepr):

            def __init__(self, b, c):
                super().__init__(b, c)

        ind = Ind('foo', 42)
        expected = "Ind:\n[ 0] 'foo'\n[ 1] 42"
        actual = self.i._name(ind)
        self.assertEqual(expected, actual)


class TestMisc(unittest.TestCase):

    class Args(IndentRepr):

        def __init__(self, *args) -> None:
            super().__init__(*args)

    def test_pickle_works(self):
        test = self.Args(f)
        _ = pickle.dumps(test)

    def test_pickle_raised_lambda(self):
        test = self.Args(lambda x: x + 1)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(test)


if __name__ == '__main__':
    unittest.main()
