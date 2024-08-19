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


class A(ArgRepr):

    def __init__(self, a):
        super().__init__(a)


class CallA(ArgRepr):

    def __init__(self, a):
        super().__init__(a)

    def __call__(self):
        pass

    @classmethod
    def c(cls):
        pass

    def m(self):
        pass

    @staticmethod
    def s():
        pass


class TestArgumentTypes(unittest.TestCase):

    class Empty(IndentRepr):

        def __init__(self) -> None:
            super().__init__()

    class Ind(IndentRepr):
        pass

    class CallInd(IndentRepr):

        def __call__(self):
            pass

        @classmethod
        def c(cls):
            pass

        def m(self):
            pass

        @staticmethod
        def s():
            pass

    def test_empty_super_init(self):
        actual = self.Empty()
        self.assertEqual('Empty()', repr(actual))

    def test_no_items(self):
        actual = self.Ind()
        self.assertEqual('Ind()', repr(actual))

    def test_int(self):
        actual = self.Ind([1, 2, 3])
        expected = "Ind():\n[ 0] 1\n[ 1] 2\n[ 2] 3"
        self.assertEqual(expected, repr(actual))

    def test_string(self):
        actual = self.Ind(['a', 'b', 'c'])
        expected = "Ind():\n[ 0] 'a'\n[ 1] 'b'\n[ 2] 'c'"
        self.assertEqual(expected, repr(actual))

    def test_custom_object(self):
        cls = Cls()
        actual = self.Ind([cls])
        expected = f"Ind():\n[ 0] {str(cls)}"
        self.assertEqual(expected, repr(actual))

    def test_argrepr(self):
        actual = self.Ind([A(1)])
        expected = "Ind():\n[ 0] A(1)"
        self.assertEqual(expected, repr(actual))

    def test_callable(self):
        cls = Cls()
        actual = self.Ind([
            f,
            Cls,
            lambda x: x,
            Call(),
            CallA(1),
            cls.m,
            Cls.c,
            cls.s,
            CallA(1).m,
            CallA.c,
            CallA(1).s,
            self.CallInd(1, 2, 3).m,
            self.CallInd.c,
            self.CallInd(1, 2, 3).s
        ])
        expected = ("Ind():\n[ 0] f\n[ 1] Cls\n[ 2] lambda\n[ 3] Call(...)\n"
                    "[ 4] CallA(1)\n[ 5] Cls.m\n[ 6] Cls.c\n[ 7] Cls.s\n"
                    "[ 8] CallA.m\n[ 9] CallA.c\n[10] CallA.s\n"
                    "[11] TestArgumentTypes.CallInd.m\n"
                    "[12] TestArgumentTypes.CallInd.c\n"
                    "[13] TestArgumentTypes.CallInd.s")
        self.assertEqual(expected, repr(actual))

    def test_nested_once(self):
        child = self.Ind([2, 3])
        parent = self.Ind([1, child, 4])
        expected_child = "[ 1] Ind():\n     [ 0] 2\n     [ 1] 3\n"
        expected = f"Ind():\n[ 0] 1\n{expected_child}[ 2] 4"
        self.assertEqual(expected, repr(parent))

    def test_nested_twice(self):
        child = self.Ind([3, 4])
        parent = self.Ind([2, child, 5])
        grand = self.Ind([1, parent, 6])
        ch = "     [ 1] Ind():\n          [ 0] 3\n          [ 1] 4\n"
        pa = f"[ 1] Ind():\n     [ 0] 2\n{ch}     [ 2] 5\n"
        expected = f"Ind():\n[ 0] 1\n{pa}[ 2] 6"
        self.assertEqual(expected, repr(grand))

    def test_callable_nested_once(self):
        child = self.CallInd([2, 3])
        parent = self.CallInd([1, child, 4])
        expected_child = "[ 1] CallInd():\n     [ 0] 2\n     [ 1] 3\n"
        expected = f"CallInd():\n[ 0] 1\n{expected_child}[ 2] 4"
        self.assertEqual(expected, repr(parent))

    def test_callable_nested_twice(self):
        child = self.CallInd([3, 4])
        parent = self.CallInd([2, child, 5])
        grand = self.CallInd([1, parent, 6])
        ch = "     [ 1] CallInd():\n          [ 0] 3\n          [ 1] 4\n"
        pa = f"[ 1] CallInd():\n     [ 0] 2\n{ch}     [ 2] 5\n"
        expected = f"CallInd():\n[ 0] 1\n{pa}[ 2] 6"
        self.assertEqual(expected, repr(grand))

    def test_empty_items_args(self):
        actual = self.Ind([], 'answer', 42)
        expected = "Ind('answer', 42)"
        self.assertEqual(expected, repr(actual))

    def test_items_args(self):
        actual = self.Ind(['question'], 'answer', 42)
        expected = "Ind('answer', 42):\n[ 0] 'question'"
        self.assertEqual(expected, repr(actual))

    def test_empty_items_kwargs(self):
        actual = self.Ind([], answer=42)
        expected = "Ind(answer=42)"
        self.assertEqual(expected, repr(actual))

    def test_items_kwargs(self):
        actual = self.Ind(['question'], answer=42)
        expected = "Ind(answer=42):\n[ 0] 'question'"
        self.assertEqual(expected, repr(actual))

    def test_empty_items_args_kwargs(self):
        actual = self.Ind([], 'question', answer=42)
        expected = "Ind('question', answer=42)"
        self.assertEqual(expected, repr(actual))

    def test_items_args_kwargs(self):
        actual = self.Ind(['everything'], 'question', answer=42)
        expected = "Ind('question', answer=42):\n[ 0] 'everything'"
        self.assertEqual(expected, repr(actual))


class TestInheritance(unittest.TestCase):

    def test_parent_indentrepr(self):

        class Parent(IndentRepr):

            def __init__(self, a):
                super().__init__([a])
                self.a = a

        class Child(Parent):

            def __init__(self, a):
                super().__init__(a)

        child = Child(1)
        self.assertEqual('Child():\n[ 0] 1', repr(child))
        self.assertTrue(hasattr(child, 'a'))
        self.assertEqual(1, child.a)

    def test_child_indentrepr_last(self):

        class Parent:

            def __init__(self, a):
                self.a = a

        class Child(Parent, IndentRepr):

            def __init__(self, a):
                super(Parent, self).__init__([a])
                super().__init__(a)

        child = Child(1)
        self.assertEqual('Child():\n[ 0] 1', repr(child))
        self.assertTrue(hasattr(child, 'a'))
        self.assertEqual(1, child.a)

    def test_child_indentrepr_first(self):

        class Parent:

            def __init__(self, a):
                self.a = a

        class Child(IndentRepr, Parent):

            def __init__(self, a):
                super(IndentRepr, self).__init__(a)
                super().__init__([a])

        child = Child(1)
        self.assertEqual('Child():\n[ 0] 1', repr(child))
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

    def test_name_of_none(self):
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
                super().__init__([b, c])

        ind = Ind('foo', 42)
        expected = "Ind():\n[ 0] 'foo'\n[ 1] 42"
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
