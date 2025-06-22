import unittest
from swak.cli import EnvParser


class TestAttributes(unittest.TestCase):

    def test_default_instantiation(self):
        _ = EnvParser()

    def test_default_has_prefix(self):
        e = EnvParser()
        self.assertTrue(hasattr(e, 'prefix'))

    def test_default_prefix_type(self):
        e = EnvParser()
        self.assertIsInstance(e.prefix, str)

    def test_custom_instantiation(self):
        _ = EnvParser('PRE_')

    def test_custom_has_prefix(self):
        e = EnvParser('PRE_')
        self.assertTrue(hasattr(e, 'prefix'))

    def test_custom_prefix_type(self):
        e = EnvParser('PRE_')
        self.assertIsInstance(e.prefix, str)

    def test_custom_prefix_value(self):
        e = EnvParser('PRE_')
        self.assertEqual('PRE_', e.prefix)

    def test_custom_prefix_stripped(self):
        e = EnvParser(' PRE_  ')
        self.assertEqual('PRE_', e.prefix)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        parse = EnvParser()
        self.assertTrue(callable(parse))

    def test_call_empty(self):
        parse = EnvParser()
        actual = parse()
        self.assertIsInstance(actual, dict)

    def test_call_with_env(self):
        inp = {'FOO': 'foo', 'BAR': 'bar'}
        parse = EnvParser()
        actual = parse(inp)
        self.assertDictEqual(inp, actual)


class TestPrefix(unittest.TestCase):

    def test_prefix_not_stripped(self):
        inp = {'FOO': 'foo', 'BAR': 'bar', 'PRE_BAR': 'baz'}
        parse = EnvParser()
        actual = parse(inp)
        self.assertDictEqual(inp, actual)

    def test_prefix_stripped(self):
        inp = {'FOO': 'foo', 'BAR': 'bar', 'PRE_BAR': 'baz'}
        parse = EnvParser('PRE_')
        actual = parse(inp)
        self.assertDictEqual({'FOO': 'foo', 'BAR': 'baz'}, actual)


class TestParse(unittest.TestCase):

    def setUp(self):
        self.parse = EnvParser()

    def test_custom_objects_unaffected(self):
        custom = object()
        inp = {'FOO': custom}
        actual = self.parse(inp)
        self.assertDictEqual(inp, actual)
        self.assertIs(actual['FOO'], custom)

    def test_quoted_string(self):
        inp = {'FOO': '"foo"'}
        actual = self.parse(inp)
        self.assertEqual('foo', actual['FOO'])

    def test_int(self):
        inp = {'FOO': '1'}
        actual = self.parse(inp)
        self.assertIsInstance(actual['FOO'], int)
        self.assertEqual(1, actual['FOO'])

    def test_float(self):
        inp = {'FOO': '1.0'}
        actual = self.parse(inp)
        self.assertIsInstance(actual['FOO'], float)
        self.assertEqual(1.0, actual['FOO'])

    def test_bool(self):
        inp = {'FOO': 'True'}
        actual = self.parse(inp)
        self.assertIsInstance(actual['FOO'], bool)
        self.assertTrue(actual['FOO'])

    def test_json_bool(self):
        inp = {'FOO': 'true'}
        actual = self.parse(inp)
        self.assertIsInstance(actual['FOO'], bool)
        self.assertTrue(actual['FOO'])

    def test_none(self):
        inp = {'FOO': 'None'}
        actual = self.parse(inp)
        self.assertIsNone(actual['FOO'])

    def test_json_none(self):
        inp = {'FOO': 'null'}
        actual = self.parse(inp)
        self.assertIsNone(actual['FOO'])

    def test_list(self):
        inp = {'FOO': '[1, 2, 3]'}
        actual = self.parse(inp)
        self.assertListEqual([1, 2, 3], actual['FOO'])

    def test_dict(self):
        inp = {'FOO': '{"foo": [1, 2, 3], "bar.baz": "Hello world!"}'}
        expected = {'foo': [1, 2, 3], 'bar.baz': 'Hello world!'}
        actual = self.parse(inp)
        self.assertDictEqual(expected, actual['FOO'])

    def test_tuple(self):
        inp = {'FOO': '(1, 2, 3)'}
        actual = self.parse(inp)
        self.assertTupleEqual((1, 2, 3), actual['FOO'])

    def test_set(self):
        inp = {'FOO': '{1, 2, 2}'}
        actual = self.parse(inp)
        self.assertSetEqual({1, 2}, actual['FOO'])

    def test_timestamp(self):
        inp = {'FOO': '2023-01-13 13:26:45.123+01:00'}
        actual = self.parse(inp)
        self.assertEqual('2023-01-13 13:26:45.123+01:00', actual['FOO'])


class TestMisc(unittest.TestCase):

    def test_repr_default(self):
        parser = EnvParser()
        self.assertEqual("EnvParser('')", repr(parser))

    def test_repr_prefix(self):
        parser = EnvParser('PREFIX_')
        self.assertEqual("EnvParser('PREFIX_')", repr(parser))


if __name__ == '__main__':
    unittest.main()
