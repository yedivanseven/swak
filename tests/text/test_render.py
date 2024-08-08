import unittest
from swak.text import TemplateRenderer


class TestEmptyInstantiation(unittest.TestCase):

    def setUp(self) -> None:
        self.empty = ''
        self.first = 'world'
        self.second = 'style'

    def test_empty(self):
        t = TemplateRenderer(self.empty)
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.empty, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_empty_dict(self):
        t = TemplateRenderer(self.empty, {})
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.empty, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_dict(self):
        t = TemplateRenderer(self.empty, {'first': self.first})
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.empty, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_kwarg(self):
        t = TemplateRenderer(self.empty, first=self.first)
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.empty, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_empty_dict_kwarg(self):
        t = TemplateRenderer(self.empty, {}, first=self.first)
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.empty, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_dict_kwarg_different_keys(self):
        t = TemplateRenderer(
            self.empty,
            {'first': self.first},
            second=self.second
        )
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.empty, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_dict_kwarg_same_keys(self):
        t = TemplateRenderer(
            self.empty,
            {'first': self.second},
            first=self.first
        )
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.empty, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)


class TestNoKeyInstantiation(unittest.TestCase):

    def setUp(self) -> None:
        self.no_key = 'Hello world!'
        self.first = 'world'
        self.second = 'style'

    def test_no_key(self):
        t = TemplateRenderer(self.no_key)
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.no_key, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_empty_dict(self):
        t = TemplateRenderer(self.no_key, {})
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.no_key, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_dict(self):
        t = TemplateRenderer(self.no_key, {'first': self.first})
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.no_key, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_kwarg(self):
        t = TemplateRenderer(self.no_key, first=self.first)
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.no_key, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_empty_dict_kwarg(self):
        t = TemplateRenderer(self.no_key, {}, first=self.first)
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.no_key, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_dict_kwarg_different_keys(self):
        t = TemplateRenderer(
            self.no_key,
            {'first': self.first},
            second=self.second
        )
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.no_key, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_dict_kwarg_same_keys(self):
        t = TemplateRenderer(
            self.no_key,
            {'first': self.second},
            first=self.first
        )
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.no_key, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)


class TestOneKeyInstantiation(unittest.TestCase):

    def setUp(self) -> None:
        self.one_key = 'Hello ${first}!'
        self.first = 'world'
        self.second = 'style'

    def test_one_key(self):
        t = TemplateRenderer(self.one_key)
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.one_key, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual(['first'], t.identifiers)

    def test_empty_dict(self):
        t = TemplateRenderer(self.one_key, {})
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.one_key, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual(['first'], t.identifiers)

    def test_right_dict(self):
        t = TemplateRenderer(self.one_key, {'first': self.first})
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual('Hello world!', t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_wrong_dict(self):
        t = TemplateRenderer(self.one_key, {'second': self.second})
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.one_key, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual(['first'], t.identifiers)

    def test_right_kwarg(self):
        t = TemplateRenderer(self.one_key, first=self.first)
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual('Hello world!', t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_wrong_kwarg(self):
        t = TemplateRenderer(self.one_key, second=self.second)
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.one_key, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual(['first'], t.identifiers)

    def test_empty_dict_right_kwarg(self):
        t = TemplateRenderer(self.one_key, {}, first=self.first)
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual('Hello world!', t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_empty_dict_wrong_kwarg(self):
        t = TemplateRenderer(self.one_key, {}, second=self.second)
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.one_key, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual(['first'], t.identifiers)

    def test_right_dict_wrong_kwarg(self):
        t = TemplateRenderer(
            self.one_key,
            {'first': self.first},
            second=self.second
        )
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual('Hello world!', t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_wrong_dict_right_kwarg(self):
        t = TemplateRenderer(
            self.one_key,
            {'second': self.second},
            first=self.first
        )
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual('Hello world!', t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)

    def test_wrong_dict_wrong_kwarg(self):
        t = TemplateRenderer(
            self.one_key,
            {'second': self.second},
            third='foo'
        )
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual(self.one_key, t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual(['first'], t.identifiers)

    def test_kwarg_overwrites_dict(self):
        t = TemplateRenderer(
            self.one_key,
            {'first': self.second},
            first=self.first
        )
        self.assertTrue(hasattr(t, 'template'))
        self.assertEqual('Hello world!', t.template)
        self.assertTrue(hasattr(t, 'identifiers'))
        self.assertListEqual([], t.identifiers)


class TestEmptyCall(unittest.TestCase):

    def setUp(self) -> None:
        self.empty = ''
        self.t = TemplateRenderer(self.empty)
        self.first = 'world'
        self.second = 'style'

    def test_callable(self):
        self.assertTrue(callable(self.t))

    def test_empty(self):
        actual = self.t()
        self.assertEqual(self.empty, actual)

    def test_empty_dict(self):
        actual = self.t({})
        self.assertEqual(self.empty, actual)

    def test_dict(self):
        actual = self.t({'first': self.first})
        self.assertEqual(self.empty, actual)

    def test_kwarg(self):
        actual = self.t(first=self.first)
        self.assertEqual(self.empty, actual)

    def test_empty_dict_kwarg(self):
        actual = self.t({}, first=self.first)
        self.assertEqual(self.empty, actual)

    def test_dict_kwarg_different_keys(self):
        actual = self.t({'first': self.first}, second=self.second)
        self.assertEqual(self.empty, actual)

    def test_dict_kwarg_same_keys(self):
        actual = self.t({'first': self.second}, first=self.first)
        self.assertEqual(self.empty, actual)


class TestNoKeyCall(unittest.TestCase):

    def setUp(self) -> None:
        self.no_key = 'Hello World!'
        self.t = TemplateRenderer(self.no_key)
        self.first = 'world'
        self.second = 'style'

    def test_callable(self):
        self.assertTrue(callable(self.t))

    def test_no_key(self):
        actual = self.t()
        self.assertEqual(self.no_key, actual)

    def test_empty_dict(self):
        actual = self.t({})
        self.assertEqual(self.no_key, actual)

    def test_dict(self):
        actual = self.t({'first': self.first})
        self.assertEqual(self.no_key, actual)

    def test_kwarg(self):
        actual = self.t(first=self.first)
        self.assertEqual(self.no_key, actual)

    def test_empty_dict_kwarg(self):
        actual = self.t({}, first=self.first)
        self.assertEqual(self.no_key, actual)

    def test_dict_kwarg_different_keys(self):
        actual = self.t({'first': self.first}, second=self.second)
        self.assertEqual(self.no_key, actual)

    def test_dict_kwarg_same_keys(self):
        actual = self.t({'first': self.second}, first=self.first)
        self.assertEqual(self.no_key, actual)


class TestOneKeyCall(unittest.TestCase):

    def setUp(self) -> None:
        self.one_key = 'Hello ${first}!'
        self.t = TemplateRenderer(self.one_key)
        self.first = 'world'
        self.second = 'style'

    def test_callable(self):
        self.assertTrue(callable(self.t))

    def test_raises_on_empty(self):
        with self.assertRaises(KeyError):
            _ = self.t()

    def test_raises_on_empty_dict(self):
        with self.assertRaises(KeyError):
            _ = self.t({})

    def test_right_dict(self):
        actual = self.t({'first': self.first})
        self.assertEqual('Hello world!', actual)

    def test_raises_on_wrong_dict(self):
        with self.assertRaises(KeyError):
            _ = self.t({'second': self.second})

    def test_right_kwarg(self):
        actual = self.t(first=self.first)
        self.assertEqual('Hello world!', actual)

    def test_raises_on_wrong_kwarg(self):
        with self.assertRaises(KeyError):
            _ = self.t(second=self.second)

    def test_empty_dict_right_kwarg(self):
        actual = self.t({}, first=self.first)
        self.assertEqual('Hello world!', actual)

    def test_raises_empty_dict_wrong_kwarg(self):
        with self.assertRaises(KeyError):
            _ = self.t({}, second=self.second)

    def test_right_dict_wrong_kwarg(self):
        actual = self.t({'first': self.first}, second=self.second)
        self.assertEqual('Hello world!', actual)

    def test_wrong_dict_right_kwarg(self):
        actual = self.t({'second': self.second}, first=self.first)
        self.assertEqual('Hello world!', actual)

    def test_raises_on_wrong_dict_wrong_kwarg(self):
        with self.assertRaises(KeyError):
            _ = self.t({'second': self.second}, second=self.second)

    def test_kwarg_overwrites_dict(self):
        actual = self.t({'first': self.second}, first=self.first)
        self.assertEqual('Hello world!', actual)


class TestTwoKeyInstantiationCall(unittest.TestCase):

    def setUp(self):
        self.two_key = 'Hello ${first}! I like your ${second}!'
        self.first = 'world'
        self.second = 'style'
        self.expected = 'Hello world! I like your style!'

    def test_instantiation_dict_call_dict(self):
        t = TemplateRenderer(self.two_key, {'first': self.first})
        actual = t({'second': self.second})
        self.assertEqual(self.expected, actual)

    def test_instantiation_kwarg_call_dict(self):
        t = TemplateRenderer(self.two_key, first=self.first)
        actual = t({'second': self.second})
        self.assertEqual(self.expected, actual)

    def test_instantiation_dict_call_kwarg(self):
        t = TemplateRenderer(self.two_key, {'first': self.first})
        actual = t(second=self.second)
        self.assertEqual(self.expected, actual)

    def test_instantiation_kwarg_call_kwarg(self):
        t = TemplateRenderer(self.two_key, first=self.first)
        actual = t(second=self.second)
        self.assertEqual(self.expected, actual)


class TestMagic(unittest.TestCase):

    def test_str_no_key(self):
        t = TemplateRenderer('Hello world!')
        self.assertEqual('Hello world!', str(t))

    def test_str_key(self):
        t = TemplateRenderer('Hello ${first}!')
        self.assertEqual('Hello ${first}!', str(t))

    def test_bool_no_key(self):
        t = TemplateRenderer('Hello world!')
        self.assertTrue(t)

    def test_bool_key(self):
        t = TemplateRenderer('Hello ${first}!')
        self.assertFalse(t)


class TestMisc(unittest.TestCase):

    def test_repr_short(self):
        t = TemplateRenderer('Hello world!')
        expected = "TemplateRenderer('Hello world!')"
        self.assertEqual(expected, repr(t))

    def test_repr_long(self):
        t = TemplateRenderer('Hello world!' * 5)
        expected = "TemplateRenderer('Hello world!Hello world!Hello ...')"
        self.assertEqual(expected, repr(t))

    def test_caps_instantiation_dict(self):
        t = TemplateRenderer('Hello ${first}!', {'First': 'world'})
        self.assertEqual('Hello ${first}!', t.template)

    def test_caps_instantiation_kwarg(self):
        t = TemplateRenderer('Hello ${first}!', FIRST='world')
        self.assertEqual('Hello ${first}!', t.template)

    def test_caps_call_dict(self):
        t = TemplateRenderer('Hello ${first}!')
        with self.assertRaises(KeyError):
            _ = t({'First': 'world'})

    def test_caps_call_kwarg(self):
        t = TemplateRenderer('Hello ${first}!')
        with self.assertRaises(KeyError):
            _ = t(FIRST='world')

    def test_non_identifier_key_instantiation(self):
        t = TemplateRenderer('Hello ${1}!', {'1': 'world'})
        self.assertEqual('Hello ${1}!', t.template)
        t = TemplateRenderer('Hello ${1}!', {1: 'world'})
        self.assertEqual('Hello ${1}!', t.template)

    def test_non_identifier_key_call(self):
        t = TemplateRenderer('Hello ${1}!')
        with self.assertRaises(ValueError):
            _ = t({'1': 'world'})
        with self.assertRaises(ValueError):
            _ = t({1: 'world'})

    def test_dot_keys_instantiation(self):
        t = TemplateRenderer('Hello ${my.dear}!', {'my.dear': 'world'})
        self.assertEqual('Hello world!', t.template)

    def test_dot_keys_call(self):
        t = TemplateRenderer('Hello ${my.dear}!')
        actual = t({'my.dear': 'world'})
        self.assertEqual('Hello world!', actual)

    def test_non_identifier_dot_key_instantiation(self):
        t = TemplateRenderer('Hello ${my.1}!', {'my.1': 'world'})
        self.assertEqual('Hello ${my.1}!', t.template)

    def test_raises_non_identifier_dot_key_call(self):
        t = TemplateRenderer('Hello ${my.1}!')
        with self.assertRaises(ValueError):
            _ = t({'my.1': 'world'})


if __name__ == '__main__':
    unittest.main()
