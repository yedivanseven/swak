import unittest
import pickle
from swak.misc import ArgRepr, IndentRepr


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


class TestSuperCallSignatures(unittest.TestCase):

    def test_minimal_class(self):

        class Test(ArgRepr):
            pass

        test = Test()
        self.assertEqual('Test()', repr(test))

    def test_empty_init_empty_super(self):

        class Test(ArgRepr):

            def __init__(self):
                super().__init__()

        test = Test()
        self.assertEqual('Test()', repr(test))

    def test_arg_empty_super(self):

        class Test(ArgRepr):

            def __init__(self, a):
                super().__init__()

        test = Test(1)
        self.assertEqual('Test()', repr(test))

    def test_kwarg_empty_super(self):

        class Test(ArgRepr):

            def __init__(self, a):
                super().__init__()

        test = Test(a=1)
        self.assertEqual('Test()', repr(test))

    def test_arg_kwarg_empty_super(self):

        class Test(ArgRepr):

            def __init__(self, a, b):
                super().__init__()

        test = Test(1, b=2)
        self.assertEqual('Test()', repr(test))

    def test_arg_init_arg_super(self):

        class Test(ArgRepr):

            def __init__(self, a):
                super().__init__(a)

        test = Test(1)
        self.assertEqual('Test(1)', repr(test))

    def test_arg_init_kwarg_super(self):

        class Test(ArgRepr):

            def __init__(self, a):
                super().__init__(a=a)

        test = Test(1)
        self.assertEqual('Test(a=1)', repr(test))

    def test_kwarg_init_arg_super(self):

        class Test(ArgRepr):

            def __init__(self, a):
                super().__init__(a)

        test = Test(a=1)
        self.assertEqual('Test(1)', repr(test))

    def test_kwarg_init_kwarg_super(self):

        class Test(ArgRepr):

            def __init__(self, a):
                super().__init__(a=a)

        test = Test(a=1)
        self.assertEqual('Test(a=1)', repr(test))

    def test_arg_kwarg_init_args_super(self):

        class Test(ArgRepr):

            def __init__(self, a, b):
                super().__init__(a, b)

        test = Test(1, b=2)
        self.assertEqual('Test(1, 2)', repr(test))

    def test_arg_kwarg_init_kwargs_super(self):

        class Test(ArgRepr):

            def __init__(self, a, b):
                super().__init__(a=a, b=b)

        test = Test(1, b=2)
        self.assertEqual('Test(a=1, b=2)', repr(test))

    def test_arg_kwarg_init_arg_kwarg_super(self):

        class Test(ArgRepr):

            def __init__(self, a, b):
                super().__init__(a, b=b)

        test = Test(1, b=2)
        self.assertEqual('Test(1, b=2)', repr(test))


class TestInheritance(unittest.TestCase):

    def test_parent_argrepr(self):

        class Parent(ArgRepr):

            def __init__(self, a):
                super().__init__(a)
                self.a = a

        class Child(Parent):

            def __init__(self, a):
                super().__init__(a)

        child = Child(1)
        self.assertEqual('Child(1)', repr(child))
        self.assertTrue(hasattr(child, 'a'))
        self.assertEqual(1, child.a)

    def test_child_argrepr_last(self):

        class Parent:

            def __init__(self, a):
                self.a = a

        class Child(Parent, ArgRepr):

            def __init__(self, a):
                super(Parent, self).__init__(a)
                super().__init__(a)

        child = Child(1)
        self.assertEqual('Child(1)', repr(child))
        self.assertTrue(hasattr(child, 'a'))
        self.assertEqual(1, child.a)

    def test_child_argrepr_first(self):

        class Parent:

            def __init__(self, a):
                self.a = a

        class Child(ArgRepr, Parent):

            def __init__(self, a):
                super(ArgRepr, self).__init__(a)
                super().__init__(a)

        child = Child(1)
        self.assertEqual('Child(1)', repr(child))
        self.assertTrue(hasattr(child, 'a'))
        self.assertEqual(1, child.a)


class TestArgumentTypes(unittest.TestCase):

    class Test(ArgRepr):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def test_argrepr_arg(self):
        test = self.Test(self.Test(1))
        self.assertEqual('Test(Test(1))', repr(test))

    def test_argrepr_kwarg(self):
        test = self.Test(b=self.Test(a=1))
        self.assertEqual('Test(b=Test(a=1))', repr(test))

    def test_indentrepr_arg(self):
        test = self.Test(Ind([1, 2, 3]))
        self.assertEqual('Test(Ind()[3])', repr(test))

    def test_indentrepr_kwarg(self):
        test = self.Test(b=Ind([1, 2, 3]))
        self.assertEqual('Test(b=Ind()[3])', repr(test))

    def test_none_arg(self):
        test = self.Test(None)
        self.assertEqual('Test(None)', repr(test))

    def test_none_kwarg(self):
        test = self.Test(a=None)
        self.assertEqual('Test(a=None)', repr(test))

    def test_object(self):
        cls = Cls()
        test = self.Test(cls)
        expected = "test_argrepr.Cls object at "
        self.assertIn(expected, repr(test))

    def test_string_arg(self):
        test = self.Test('1')
        self.assertEqual("Test('1')", repr(test))
        test = self.Test("1")
        self.assertEqual("Test('1')", repr(test))

    def test_string_kwarg(self):
        test = self.Test(a='1')
        self.assertEqual("Test(a='1')", repr(test))
        test = self.Test(a="1")
        self.assertEqual("Test(a='1')", repr(test))


class TestCallables(unittest.TestCase):

    class Test(ArgRepr):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

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

    def test_lambda(self):
        test = self.Test(lambda x: x + 1)
        self.assertEqual('Test(lambda)', repr(test))

    def test_function(self):
        test = self.Test(f)
        self.assertEqual('Test(f)', repr(test))

    def test_method(self):
        cls = Cls()
        test = self.Test(cls.m)
        self.assertEqual('Test(Cls.m)', repr(test))

    def test_classmethod(self):
        test = self.Test(Cls.c)
        self.assertEqual('Test(Cls.c)', repr(test))

    def test_staticmethod(self):
        cls = Cls()
        test = self.Test(cls.s)
        self.assertEqual('Test(Cls.s)', repr(test))

    def test_class(self):
        test = self.Test(Cls)
        self.assertEqual('Test(Cls)', repr(test))

    def test_callable_object(self):
        cls = Call()
        test = self.Test(cls)
        self.assertEqual('Test(Call(...))', repr(test))

    def test_callable_argrepr(self):
        test = self.Test(self.Test(1))
        self.assertEqual('Test(Test(1))', repr(test))

    def test_argrepr_method(self):
        test = self.Test(self.Test(1, 2, 3).m)
        self.assertEqual('Test(TestCallables.Test.m)', repr(test))

    def test_argrepr_classmethod(self):
        test = self.Test(self.Test.c)
        self.assertEqual('Test(TestCallables.Test.c)', repr(test))

    def test_argrepr_staticmethod(self):
        test = self.Test(self.Test(1, 2, 3).s)
        self.assertEqual('Test(TestCallables.Test.s)', repr(test))

    def test_callable_indentrepr(self):
        test = self.Test(CallInd([1, 2, 3]))
        self.assertEqual('Test(CallInd()[3])', repr(test))

    def test_indentrepr_method(self):
        test = self.Test(CallInd([1, 2, 3]).m)
        self.assertEqual('Test(CallInd.m)', repr(test))

    def test_indentrepr_classmethod(self):
        test = self.Test(CallInd.c)
        self.assertEqual('Test(CallInd.c)', repr(test))

    def test_indentrepr_staticmethod(self):
        test = self.Test(CallInd([1, 2, 3]).s)
        self.assertEqual('Test(CallInd.s)', repr(test))


class TestMisc(unittest.TestCase):

    class Test(ArgRepr):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def test_pickle_works(self):
        test = self.Test(f)
        _ = pickle.loads(pickle.dumps(test))

    def test_pickle_raises_with_lambda(self):
        test = self.Test(lambda x: x + 1)
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(test))


if __name__ == '__main__':
    unittest.main()
