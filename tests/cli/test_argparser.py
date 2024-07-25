import unittest
from argparse import RawTextHelpFormatter, RawDescriptionHelpFormatter
from unittest.mock import patch
from swak.cli import ArgParser, USAGE, DESCRIPTION, EPILOG
from swak.cli.exceptions import ArgParseError


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = ArgParser()

    # ToDo: Test that Argparse constructor is called!

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
        self.assertIsInstance(parse.usage, str)
        self.assertEqual(USAGE, parse.usage)

    def test_has_description(self):
        parse = ArgParser()
        self.assertTrue(hasattr(parse, 'description'))

    def test_description(self):
        parse = ArgParser()
        self.assertIsInstance(parse.description, str)
        self.assertEqual(DESCRIPTION, parse.description)

    def test_has_epilog(self):
        parse = ArgParser()
        self.assertTrue(hasattr(parse, 'epilog'))

    def test_epilog(self):
        parse = ArgParser()
        self.assertIsInstance(parse.epilog, str)
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

# ToDo: Continue here by repeating default attribute section!
# ToDo: Test that Argparse constructor is called!


class TestInstantiation(unittest.TestCase):

    def test_defaults(self):
        parse = ArgParser()
        self.assertTrue(hasattr(parse, 'default_action'))
        self.assertIsNone(parse.default_action)
        self.assertTrue(hasattr(parse, 'usage'))
        self.assertIsInstance(parse.usage, str)
        self.assertEqual(USAGE, parse.usage)
        self.assertTrue(hasattr(parse, 'description'))
        self.assertIsInstance(parse.description, str)
        self.assertEqual(DESCRIPTION, parse.description)
        self.assertTrue(hasattr(parse, 'epilog'))
        self.assertIsInstance(parse.epilog, str)
        self.assertEqual(EPILOG, parse.epilog)
        self.assertTrue(hasattr(parse, 'fmt_cls'))
        self.assertIs(RawTextHelpFormatter, parse.fmt_cls)
        self.assertTrue(hasattr(parse, 'kwargs'))
        self.assertDictEqual({}, parse.kwargs)

    def test_repr_defaults(self):
        parse = ArgParser()
        self.assertEqual("ArgParser(None)", repr(parse))

    def test_args_kwargs(self):
        parse = ArgParser('action', 'foo', 'bar', 'baz', prog='foo')
        self.assertTrue(hasattr(parse, 'default_action'))
        self.assertIsInstance(parse.default_action, str)
        self.assertEqual('action', parse.default_action)
        self.assertTrue(hasattr(parse, 'usage'))
        self.assertIsInstance(parse.usage, str)
        self.assertEqual('foo', parse.usage)
        self.assertTrue(hasattr(parse, 'description'))
        self.assertIsInstance(parse.description, str)
        self.assertEqual('bar', parse.description)
        self.assertTrue(hasattr(parse, 'epilog'))
        self.assertIsInstance(parse.epilog, str)
        self.assertEqual('baz', parse.epilog)
        self.assertTrue(hasattr(parse, 'kwargs'))
        self.assertDictEqual({'prog': 'foo'}, parse.kwargs)

    def test_repr_args_kwargs(self):
        parse = ArgParser('action', prog='foo')
        expected = "ArgParser('action', prog='foo')"
        self.assertEqual(expected, repr(parse))

    def test_allow_abbrev_kwarg_popped(self):
        parse = ArgParser('action', prog='foo', allow_abbrev=True)
        self.assertDictEqual({'prog': 'foo'}, parse.kwargs)
        expected = "ArgParser('action', prog='foo')"
        self.assertEqual(expected, repr(parse))

    @patch('swak.cli.argparser.ArgumentParser')
    def test_argument_parser_instantiated(self, parser):
        _ = ArgParser(
            'action',
            'foo',
            prog='foo',
            allow_abbrev=True,
            epilog='bar'
        )
        parser.assert_called_once()
        parser.assert_called_once_with(
            allow_abbrev=False,
            usage='foo',
            description=DESCRIPTION,
            epilog='bar',
            formatter_class=RawTextHelpFormatter,
            prog='foo'
        )

    def test_callable(self):
        parse = ArgParser('action', prog='foo', allow_abbrev=True)
        self.assertTrue(callable(parse))


class TestCall(unittest.TestCase):

    def test_default(self):
        parse = ArgParser()
        actions, kwargs = parse('')
        self.assertTupleEqual((), actions)
        self.assertDictEqual({}, kwargs)

    def test_action(self):
        parse = ArgParser()
        actions, kwargs = parse(['action'])
        self.assertTupleEqual(('action', ), actions)
        self.assertDictEqual({}, kwargs)

    def test_actions(self):
        parse = ArgParser()
        actions, kwargs = parse(['action', 'next'])
        self.assertTupleEqual(('action', 'next'), actions)
        self.assertDictEqual({}, kwargs)

    def test_action_underscore(self):
        parse = ArgParser()
        actions, kwargs = parse(['do_stuff', 'more_stuff'])
        self.assertTupleEqual(('do_stuff', 'more_stuff'), actions)
        self.assertDictEqual({}, kwargs)

    def test_action_hyphen(self):
        parse = ArgParser()
        actions, kwargs = parse(['do-stuff', 'more-stuff'])
        self.assertTupleEqual(('do_stuff', 'more_stuff'), actions)
        self.assertDictEqual({}, kwargs)

    def test_kwargs(self):
        parse = ArgParser()
        actions, kwargs = parse(['--foo', 'bar', '--baz', '42'])
        self.assertTupleEqual((), actions)
        self.assertDictEqual({'foo': 'bar', 'baz': '42'}, kwargs)

    def test_kwargs_equal(self):
        parse = ArgParser()
        actions, kwargs = parse(['--foo=bar', '--baz=42'])
        self.assertTupleEqual((), actions)
        self.assertDictEqual({'foo': 'bar', 'baz': '42'}, kwargs)

    def test_kwargs_timestamp(self):
        parse = ArgParser()
        actions, kwargs = parse(['--time', '2023-01-13 13:26:45.123+01:00'])
        self.assertTupleEqual((), actions)
        self.assertDictEqual({'time': '2023-01-13 13:26:45.123+01:00'}, kwargs)

    def test_kwargs_dot(self):
        parse = ArgParser()
        actions, kwargs = parse(['--foo.one', 'bar', '--baz.two', '42'])
        self.assertTupleEqual((), actions)
        self.assertDictEqual({'foo.one': 'bar', 'baz.two': '42'}, kwargs)

    def test_kwargs_hyphen(self):
        parse = ArgParser()
        actions, kwargs = parse(['--foo-one', 'bar', '--baz-two', '42'])
        self.assertTupleEqual((), actions)
        self.assertDictEqual({'foo_one': 'bar', 'baz_two': '42'}, kwargs)

    def test_kwargs_dots_and_hyphen(self):
        parse = ArgParser()
        actions, kwargs = parse(['--f.o.o-one', 'bar', '--b.a.z-two', '42'])
        self.assertTupleEqual((), actions)
        self.assertDictEqual({'f.o.o_one': 'bar', 'b.a.z_two': '42'}, kwargs)

    def test_action_and_kwargs(self):
        parse = ArgParser()
        actions, kwargs = parse(['do-stuff', '--foo.baz-one', 'bar'])
        self.assertTupleEqual(('do_stuff',), actions)
        self.assertDictEqual({'foo.baz_one': 'bar'}, kwargs)

    def test_actions_and_kwargs(self):
        parse = ArgParser()
        actions, kwargs = parse(['do-stuff', 'more_stuff', '--foo', 'bar'])
        self.assertTupleEqual(('do_stuff', 'more_stuff'), actions)
        self.assertDictEqual({'foo': 'bar'}, kwargs)

    def test_ignores_initial_abbrevs(self):
        parse = ArgParser()
        actions, kwargs = parse(['do-stuff', '-f', '-b'])
        self.assertTupleEqual(('do_stuff',), actions)
        self.assertDictEqual({}, kwargs)

    def test_ignores_initial_abbrev_no_value(self):
        parse = ArgParser()
        actions, kwargs = parse(['do-stuff', '-f', '--foo', 'bar'])
        self.assertTupleEqual(('do_stuff',), actions)
        self.assertDictEqual({'foo': 'bar'}, kwargs)

    def test_ignores_initial_abbrev_value(self):
        parse = ArgParser()
        actions, kwargs = parse(['do-stuff', '-f', 'baz', '--foo', 'bar'])
        self.assertTupleEqual(('do_stuff',), actions)
        self.assertDictEqual({'foo': 'bar'}, kwargs)

    def test_raises_on_non_initial_abbrev(self):
        parse = ArgParser()
        with self.assertRaises(ArgParseError):
            _ = parse(['do-stuff', '--foo', 'bar', '-f', 'bar'])

    def test_raises_on_invalid_action(self):
        parse = ArgParser()
        with self.assertRaises(ArgParseError):
            _ = parse(['1action', '-f', 'bar'])

    def test_raises_on_no_value(self):
        parse = ArgParser()
        with self.assertRaises(ArgParseError):
            _ = parse(['--foo', 'bar', '--baz'])

    def test_raises_on_too_many_values(self):
        parse = ArgParser()
        with self.assertRaises(ArgParseError):
            _ = parse(['--foo', 'bar', 'baz'])

    def test_raises_on_missing_value(self):
        parse = ArgParser()
        with self.assertRaises(ArgParseError):
            _ = parse(['--foo', '--bar', 'baz'])

    def test_action_raises_on_no_value(self):
        parse = ArgParser()
        with self.assertRaises(ArgParseError):
            _ = parse(['action', '--foo', 'bar', '--baz'])

    def test_action_raises_on_too_many_values(self):
        parse = ArgParser()
        with self.assertRaises(ArgParseError):
            _ = parse(['action', '--foo', 'bar', 'baz'])

    def test_action_raises_on_missing_value(self):
        parse = ArgParser()
        with self.assertRaises(ArgParseError):
            _ = parse(['action', '--foo', '--bar', 'baz'])


if __name__ == '__main__':
    unittest.main()
