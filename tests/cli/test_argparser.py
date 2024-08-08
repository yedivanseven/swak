import unittest
from unittest.mock import patch
from argparse import RawTextHelpFormatter, RawDescriptionHelpFormatter
from swak.cli import ArgParser, USAGE, DESCRIPTION, EPILOG
from swak.cli.exceptions import ArgParseError


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = ArgParser()

    @patch('swak.cli.argparser.ArgumentParser')
    def test_constructor_called(self, mock):
        _ = ArgParser()
        mock.assert_called_once()

    @patch('swak.cli.argparser.ArgumentParser')
    def test_constructor_called_with_defaults(self, mock):
        _ = ArgParser()
        mock.assert_called_once_with(
            usage=USAGE,
            description=DESCRIPTION,
            epilog=EPILOG,
            formatter_class=RawTextHelpFormatter
        )

    def test_has_default_action(self):
        parse = ArgParser()
        self.assertTrue(hasattr(parse, 'default_action'))

    def test_default_action(self):
        parse = ArgParser()
        self.assertIsNone(parse.default_action)

    def test_has_usage(self):
        parse = ArgParser()
        self.assertTrue(hasattr(parse, 'usage'))

    def test_usage(self):
        parse = ArgParser()
        self.assertEqual(USAGE, parse.usage)

    def test_has_description(self):
        parse = ArgParser()
        self.assertTrue(hasattr(parse, 'description'))

    def test_description(self):
        parse = ArgParser()
        self.assertEqual(DESCRIPTION, parse.description)

    def test_has_epilog(self):
        parse = ArgParser()
        self.assertTrue(hasattr(parse, 'epilog'))

    def test_epilog(self):
        parse = ArgParser()
        self.assertEqual(EPILOG, parse.epilog)

    def test_has_fmt_cls(self):
        parse = ArgParser()
        self.assertTrue(hasattr(parse, 'fmt_cls'))

    def test_fmt_cls(self):
        parse = ArgParser()
        self.assertIs(RawTextHelpFormatter, parse.fmt_cls)


class TestCustomAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = ArgParser(
            'action',
            'foo',
            'bar',
            'baz',
            RawDescriptionHelpFormatter
        )

    @patch('swak.cli.argparser.ArgumentParser')
    def test_constructor_called(self, mock):
        _ = ArgParser(
            'action',
            'foo',
            'bar',
            'baz',
            RawDescriptionHelpFormatter
        )
        mock.assert_called_once()

    @patch('swak.cli.argparser.ArgumentParser')
    def test_constructor_called_with_defaults(self, mock):
        _ = ArgParser(
            'action',
            'foo',
            'bar',
            'baz',
            RawDescriptionHelpFormatter
        )
        mock.assert_called_once_with(
            usage='foo',
            description='bar',
            epilog='baz',
            formatter_class=RawDescriptionHelpFormatter
        )

    def test_has_default_action(self):
        parse = ArgParser('action')
        self.assertTrue(hasattr(parse, 'default_action'))

    def test_default_action(self):
        parse = ArgParser('action')
        self.assertEqual('action', parse.default_action)

    def test_has_usage(self):
        parse = ArgParser('action', 'foo')
        self.assertTrue(hasattr(parse, 'usage'))

    def test_usage(self):
        parse = ArgParser('action', 'foo')
        self.assertEqual('foo', parse.usage)

    def test_has_description(self):
        parse = ArgParser('action', 'foo', 'bar')
        self.assertTrue(hasattr(parse, 'description'))

    def test_description(self):
        parse = ArgParser('action', 'foo', 'bar')
        self.assertEqual('bar', parse.description)

    def test_has_epilog(self):
        parse = ArgParser('action', 'foo', 'bar', 'baz')
        self.assertTrue(hasattr(parse, 'epilog'))

    def test_epilog(self):
        parse = ArgParser('action', 'foo', 'bar', 'baz')
        self.assertEqual('baz', parse.epilog)

    def test_has_fmt_cls(self):
        parse = ArgParser(
            'action',
            'foo',
            'bar',
            'baz',
            RawDescriptionHelpFormatter
        )
        self.assertTrue(hasattr(parse, 'fmt_cls'))

    def test_fmt_cls(self):
        parse = ArgParser(
            'action',
            'foo',
            'bar',
            'baz',
            RawDescriptionHelpFormatter
        )
        self.assertIs(RawDescriptionHelpFormatter, parse.fmt_cls)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.parse = ArgParser()

    def test_callable(self):
        self.assertTrue(callable(self.parse))

    def test_empty(self):
        actual = self.parse([])
        expected = (
            [],
            {}
        )
        self.assertTupleEqual(expected, actual)

    def test_action(self):
        actual = self.parse(['action'])
        expected = (
            ['action'],
            {}
        )
        self.assertTupleEqual(expected, actual)

    def test_actions(self):
        actual = self.parse(['one', 'two'])
        expected = (
            ['one', 'two'],
            {}
        )
        self.assertTupleEqual(expected, actual)

    def test_actions_underscore(self):
        actual = self.parse(['do_something', 'do_more'])
        expected = (
            ['do_something', 'do_more'],
            {}
        )
        self.assertTupleEqual(expected, actual)

    def test_actions_hyphen(self):
        actual = self.parse(['do-something', 'do-more'])
        expected = (
            ['do_something', 'do_more'],
            {}
        )
        self.assertTupleEqual(expected, actual)

    def test_actions_mixed(self):
        actual = self.parse(['do-some_thing', 'do_more-stuff'])
        expected = (
            ['do_some_thing', 'do_more_stuff'],
            {}
        )
        self.assertTupleEqual(expected, actual)

    def test_raises_not_identifier(self):
        expected = 'Actions must be valid identifiers, unlike "123_done"!'
        with self.assertRaises(ArgParseError) as error:
            _ = self.parse(['123-done'])
        self.assertEqual(expected, str(error.exception))

    def test_options_space(self):
        actual = self.parse(['--foo', 'hello', '--bar', 'world'])
        expected = (
            [],
            {
                'foo': 'hello',
                'bar': 'world'
            }
        )
        self.assertTupleEqual(expected, actual)

    def test_options_equal(self):
        actual = self.parse(['--foo=hello', '--bar=world'])
        expected = (
            [],
            {
                'foo': 'hello',
                'bar': 'world'
            }
        )
        self.assertTupleEqual(expected, actual)

    def test_options_dot(self):
        actual = self.parse(['--foo.bar', 'hello', '--bar.baz', 'world'])
        expected = (
            [],
            {
                'foo.bar': 'hello',
                'bar.baz': 'world'
            }
        )
        self.assertTupleEqual(expected, actual)

    def test_options_hyphen(self):
        actual = self.parse(['--foo-bar', 'hello', '--bar-baz', 'world'])
        expected = (
            [],
            {
                'foo_bar': 'hello',
                'bar_baz': 'world'
            }
        )
        self.assertTupleEqual(expected, actual)

    def test_actions_and_options(self):
        actual = self.parse(['one-1', 'two_2', '--foo.bar-baz', 'hello world'])
        expected = (
            ['one_1', 'two_2'],
            {
                'foo.bar_baz': 'hello world'
            }
        )
        self.assertTupleEqual(expected, actual)

    def test_ignore_short_only_no_values(self):
        actual = self.parse(['-f', '-b'])
        expected = (
            [],
            {}
        )
        self.assertTupleEqual(expected, actual)

    def test_ignore_actions_short_only(self):
        actual = self.parse(['one', 'two', '-f', '-b'])
        expected = (
            ['one', 'two'],
            {}
        )
        self.assertTupleEqual(expected, actual)

    def test_ignore_action_initial_short_no_value(self):
        actual = self.parse(['one', '-f', '--bar', 'baz'])
        expected = (
            ['one',],
            {'bar': 'baz'}
        )
        self.assertTupleEqual(expected, actual)

    def test_ignore_action_initial_short_value(self):
        actual = self.parse(['one', '-f', 'hello', '--bar', 'baz'])
        expected = (
            ['one', ],
            {'bar': 'baz'}
        )
        self.assertTupleEqual(expected, actual)

    def test_raises_on_central_short_no_value(self):
        with self.assertRaises(ArgParseError):
            _ = self.parse(['one', '--foo', 'hello', '-f', '--bar', 'baz'])

    def test_raises_on_central_short_value(self):
        with self.assertRaises(ArgParseError):
            _ = self.parse(['--foo', 'hello', '-f', 'world', '--bar', 'baz'])

    def test_raises_on_trailing_short_no_value(self):
        with self.assertRaises(ArgParseError):
            _ = self.parse(['one', '--foo', 'hello', '-f'])

    def test_raises_on_trailing_short_value(self):
        with self.assertRaises(ArgParseError):
            _ = self.parse(['one', '--foo', 'hello', '-f', 'world'])

    def test_raises_on_no_value(self):
        with self.assertRaises(ArgParseError):
            _ = self.parse(['--foo', 'bar', '--baz'])

    def test_raises_on_too_many_values(self):
        with self.assertRaises(ArgParseError):
            _ = self.parse(['--foo', 'bar', 'baz'])

    def test_raises_on_missing_value(self):
        with self.assertRaises(ArgParseError):
            _ = self.parse(['--foo', '--bar', 'baz'])

    def test_action_raises_on_no_value(self):
        with self.assertRaises(ArgParseError):
            _ = self.parse(['action', '--foo', 'bar', '--baz'])

    def test_action_raises_on_too_many_values(self):
        with self.assertRaises(ArgParseError):
            _ = self.parse(['action', '--foo', 'bar', 'baz'])

    def test_action_raises_on_missing_value(self):
        with self.assertRaises(ArgParseError):
            _ = self.parse(['action', '--foo', '--bar', 'baz'])


class TestCastValues(unittest.TestCase):

    def setUp(self):
        self.parse = ArgParser()

    def test_timestamp(self):
        expected = '2023-01-13 13:26:45.123+01:00'
        _, option = self.parse(['--key', '2023-01-13 13:26:45.123+01:00'])
        self.assertEqual(expected, option['key'])

    def test_quoted_string(self):
        expected = 'Hello world!'
        _, option = self.parse(['--key', '"Hello world!"'])
        self.assertEqual(expected, option['key'])

    def test_int(self):
        expected = 42
        _, option = self.parse(['--key', '42'])
        self.assertIsInstance(option['key'], int)
        self.assertEqual(expected, option['key'])

    def test_float(self):
        expected = 42.0
        _, option = self.parse(['--key', '42.0'])
        self.assertIsInstance(option['key'], float)
        self.assertEqual(expected, option['key'])

    def test_bool(self):
        _, option = self.parse(['--key', 'True'])
        self.assertIsInstance(option['key'], bool)
        self.assertTrue(option['key'])

    def test_json_bool(self):
        _, option = self.parse(['--key', 'false'])
        self.assertIsInstance(option['key'], bool)
        self.assertFalse(option['key'])

    def test_none(self):
        _, option = self.parse(['--key', 'None'])
        self.assertIsNone(option['key'])

    def test_json_none(self):
        _, option = self.parse(['--key', 'null'])
        self.assertIsNone(option['key'])

    def test_list(self):
        expected = [1, 2, 3]
        _, option = self.parse(['--key', '[1, 2, 3]'])
        self.assertListEqual(expected, option['key'])

    def test_dict(self):
        expected = {'foo': 1, 'bar': 2, 'baz': 3}
        _, option = self.parse(['--key', "{'foo': 1, 'bar': 2, 'baz': 3}"])
        self.assertDictEqual(expected, option['key'])

    def test_tuple(self):
        expected = 1, 2, 3
        _, option = self.parse(['--key', '(1, 2, 3)'])
        self.assertTupleEqual(expected, option['key'])

    def test_set(self):
        expected = {1, 2}
        _, option = self.parse(['--key', '{1, 2, 2}'])
        self.assertSetEqual(expected, option['key'])


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        parser = ArgParser()
        expected = 'ArgParser(None, ...)'
        self.assertEqual(expected, repr(parser))

    def test_custom_repr(self):
        parser = ArgParser(
            'action',
            'foo',
            'bar',
            'baz',
            RawDescriptionHelpFormatter
        )
        expected = 'ArgParser(action, ...)'
        self.assertEqual(expected, repr(parser))


if __name__ == '__main__':
    unittest.main()
