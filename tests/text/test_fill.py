import unittest
from swak.text import FormFiller


class TestAttributes(unittest.TestCase):

    def test_empty_instantiation(self):
        _ = FormFiller()

    def test_empty_has_mapping(self):
        f = FormFiller()
        self.assertTrue(hasattr(f, 'mapping'))

    def test_empty_mapping_correct(self):
        f = FormFiller()
        self.assertDictEqual({}, f.mapping)

    def test_empty_dict_instantiation(self):
        _ = FormFiller({})

    def test_empty_dict_has_mapping(self):
        f = FormFiller({})
        self.assertTrue(hasattr(f, 'mapping'))

    def test_empty_dict_mapping_correct(self):
        f = FormFiller({})
        self.assertDictEqual({}, f.mapping)

    def test_empty_dict_kwarg_instantiation(self):
        _ = FormFiller({}, foo='bar')

    def test_empty_dict_kwarg_has_mapping(self):
        f = FormFiller({}, foo='bar')
        self.assertTrue(hasattr(f, 'mapping'))

    def test_empty_dict_kwarg_mapping_correct(self):
        f = FormFiller({}, foo='bar')
        self.assertDictEqual({'foo': 'bar'}, f.mapping)

    def test_dict(self):
        f = FormFiller({'foo': 'bar'})
        self.assertTrue(hasattr(f, 'mapping'))
        self.assertDictEqual({'foo': 'bar'}, f.mapping)

    def test_dict_kwarg_different_keys(self):
        f = FormFiller({'foo': 'bar'}, baz=42)
        self.assertTrue(hasattr(f, 'mapping'))
        self.assertDictEqual({'foo': 'bar', 'baz': 42}, f.mapping)

    def test_kwarg_overwrites_dict(self):
        f = FormFiller({'foo': 'bar'}, foo='baz')
        self.assertTrue(hasattr(f, 'mapping'))
        self.assertDictEqual({'foo': 'baz'}, f.mapping)


class TestNoKeyUsage(unittest.TestCase):

    def setUp(self):
        self.text = 'Hello world!'

    def test_callable(self):
        f = FormFiller({'foo': 'bar'}, foo='baz')
        self.assertTrue(callable(f))

    def test_empty(self):
        f = FormFiller()
        actual = f(self.text)
        self.assertEqual(self.text, actual)

    def test_empty_dict(self):
        f = FormFiller({})
        actual = f(self.text)
        self.assertEqual(self.text, actual)

    def test_empty_dict_kwarg(self):
        f = FormFiller({}, foo='bar')
        actual = f(self.text)
        self.assertEqual(self.text, actual)

    def test_dict_kwarg(self):
        f = FormFiller({'foo': 'bar'}, baz=42)
        actual = f(self.text)
        self.assertEqual(self.text, actual)

    def test_call_empty_dict(self):
        f = FormFiller({'foo': 'bar'}, foo='baz')
        actual = f(self.text, {})
        self.assertEqual(self.text, actual)

    def test_call_empty_dict_kwarg(self):
        f = FormFiller({'foo': 'bar'}, foo='baz')
        actual = f(self.text, {}, answer=42)
        self.assertEqual(self.text, actual)

    def test_call_dict_kwarg(self):
        f = FormFiller({'foo': 'bar'}, foo='baz')
        actual = f(self.text, {'question': 'unknown'}, answer=42)
        self.assertEqual(self.text, actual)


class TestKeyUsage(unittest.TestCase):

    def setUp(self):
        self.text = 'Hello ${first}! I like your ${second}!'
        self.first = 'world'
        self.second = 'style'
        self.expected = f'Hello {self.first}! I like your {self.second}!'

    def test_instantiation_dict(self):
        f = FormFiller({'first': self.first, 'second': self.second})
        actual = f(self.text)

        self.assertEqual(self.expected, actual)

    def test_instantiation_dict_kwarg(self):
        f = FormFiller({'first': self.first}, second=self.second)
        actual = f(self.text)
        self.assertEqual(self.expected, actual)

    def test_instantiation_kwargs(self):
        f = FormFiller(first=self.first, second=self.second)
        actual = f(self.text)
        self.assertEqual(self.expected, actual)

    def test_instantiation_kwarg_overwrites_dict(self):
        f = FormFiller({'first': self.first, 'second': 42}, second=self.second)
        actual = f(self.text)
        self.assertEqual(self.expected, actual)

    def test_instantiation_dict_call_empty_dict(self):
        f = FormFiller({'first': self.first, 'second': self.second})
        actual = f(self.text, {})
        self.assertEqual(self.expected, actual)

    def test_instantiation_dict_kwarg_call_empty_dict(self):
        f = FormFiller({'first': self.first}, second=self.second)
        actual = f(self.text, {})
        self.assertEqual(self.expected, actual)

    def test_instantiation_kwargs_call_empty_dict(self):
        f = FormFiller(first=self.first, second=self.second)
        actual = f(self.text, {})
        self.assertEqual(self.expected, actual)

    def test_instantiation_kwarg_overwrites_dict_call_empty_dict(self):
        f = FormFiller({'first': self.first, 'second': 42}, second=self.second)
        actual = f(self.text, {})
        self.assertEqual(self.expected, actual)

    def test_instantiation_empty_call_dict(self):
        f = FormFiller()
        actual = f(self.text, {'first': self.first, 'second': self.second})
        self.assertEqual(self.expected, actual)

    def test_instantiation_empty_call_dict_kwarg(self):
        f = FormFiller()
        actual = f(self.text, {'first': self.first}, second=self.second)
        self.assertEqual(self.expected, actual)

    def test_instantiation_empty_call_kwargs(self):
        f = FormFiller()
        actual = f(self.text, first=self.first, second=self.second)
        self.assertEqual(self.expected, actual)

    def test_instantiation_empty_call_kwarg_overwrites_dict(self):
        f = FormFiller()
        actual = f(
            self.text,
            {'first': self.first, 'second': 'nose'},
            second=self.second
        )
        self.assertEqual(self.expected, actual)

    def test_instantiation_empty_dict_call_dict(self):
        f = FormFiller({})
        actual = f(self.text, {'first': self.first, 'second': self.second})
        self.assertEqual(self.expected, actual)

    def test_instantiation_empty_dict_call_dict_kwarg(self):
        f = FormFiller({})
        actual = f(self.text, {'first': self.first}, second=self.second)
        self.assertEqual(self.expected, actual)

    def test_instantiation_empty_dict_call_kwargs(self):
        f = FormFiller({})
        actual = f(self.text, first=self.first, second=self.second)
        self.assertEqual(self.expected, actual)

    def test_instantiation_empty_dict_call_kwarg_overwrites_dict(self):
        f = FormFiller({})
        actual = f(
            self.text,
            {'first': self.first, 'second': 'nose'},
            second=self.second
        )
        self.assertEqual(self.expected, actual)

    def test_instantiation_dict_call_dict(self):
        f = FormFiller({'first': self.first})
        actual = f(self.text, {'second': self.second})
        self.assertEqual(self.expected, actual)

    def test_instantiation_dict_call_kwarg(self):
        f = FormFiller({'first': self.first})
        actual = f(self.text, second=self.second)
        self.assertEqual(self.expected, actual)

    def test_instantiation_kwarg_call_dict(self):
        f = FormFiller(first=self.first)
        actual = f(self.text, {'second': self.second})
        self.assertEqual(self.expected, actual)

    def test_call_dict_overwrites_instantiation_dict(self):
        f = FormFiller({'first': self.first, 'second': 'nose'})
        actual = f(self.text, {'second': self.second})
        self.assertEqual(self.expected, actual)

    def test_call_dict_overwrites_instantiation_kwarg(self):
        f = FormFiller(first=self.first, second='nose')
        actual = f(self.text, {'second': self.second})
        self.assertEqual(self.expected, actual)

    def test_call_kwarg_overwrites_instantiation_dict(self):
        f = FormFiller({'first': self.first, 'second': 'nose'})
        actual = f(self.text, second=self.second)
        self.assertEqual(self.expected, actual)

    def test_call_kwarg_overwrites_instantiation_kwarg(self):
        f = FormFiller(first=self.first, second='nose')
        actual = f(self.text, second=self.second)
        self.assertEqual(self.expected, actual)

    def test_raises(self):
        f = FormFiller(first=self.first)
        with self.assertRaises(KeyError):
            _ = f(self.text, answer=42)


class TestMagic(unittest.TestCase):

    def test_bool_empty(self):
        f = FormFiller()
        self.assertFalse(f)

    def test_bool_empty_dict(self):
        f = FormFiller({})
        self.assertFalse(f)

    def test_bool_dict(self):
        f = FormFiller({'first': 'world'})
        self.assertTrue(f)

    def test_bool_kwarg(self):
        f = FormFiller(first='world')
        self.assertTrue(f)

    def test_bool_dict_kwarg(self):
        f = FormFiller({'first': 'world'}, second='style')
        self.assertTrue(f)


class TestMisc(unittest.TestCase):

    def test_repr_short(self):
        f = FormFiller({'first': 'world'})
        expected = "FormFiller({'first': 'world'})"
        self.assertEqual(expected, repr(f))

    def test_repr_long(self):
        f = FormFiller({'first': 'world', 'second': 'style'})
        expected = "FormFiller({'first': 'world', 'second': ' ...})"
        self.assertEqual(expected, repr(f))

    def test_caps_instantiation_dict(self):
        f = FormFiller({'First': 'world'})
        with self.assertRaises(KeyError):
            _ = f('Hello ${first}!')

    def test_caps_instantiation_kwarg(self):
        f = FormFiller(FIRST='world')
        with self.assertRaises(KeyError):
            _ = f('Hello ${first}!')

    def test_caps_call_dict(self):
        f = FormFiller()
        with self.assertRaises(KeyError):
            _ = f('Hello ${first}!', {'First': 'world'})

    def test_caps_call_kwarg(self):
        f = FormFiller()
        with self.assertRaises(KeyError):
            _ = f('Hello ${first}!', FIRST='world')

    def test_raises_non_identifier_key_instantiation(self):
        f = FormFiller({'1': 'world'})
        with self.assertRaises(ValueError):
            _ = f('Hello ${1}!')

    def test_raises_non_identifier_key_call(self):
        f = FormFiller()
        with self.assertRaises(ValueError):
            _ = f('Hello ${1}!', {'1': 'world'})

    def test_dot_keys_instantiation(self):
        f = FormFiller({'my.dear': 'world'})
        actual = f('Hello ${my.dear}!')
        self.assertEqual('Hello world!', actual)

    def test_dot_keys_call(self):
        f = FormFiller()
        actual = f('Hello ${my.dear}!', {'my.dear': 'world'})
        self.assertEqual('Hello world!', actual)

    def test_raises_non_identifier_dot_key_instantiation(self):
        f = FormFiller({'my.1': 'world'})
        with self.assertRaises(ValueError):
            _ = f('Hello ${my.1}!')

    def test_raises_non_identifier_dot_key_call(self):
        f = FormFiller()
        with self.assertRaises(ValueError):
            _ = f('Hello ${my.1}!', {'my.1': 'world'})


if __name__ == '__main__':
    unittest.main()
