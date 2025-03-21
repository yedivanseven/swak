import unittest
from unittest.mock import patch
from swak.misc import NotFound
from swak.text import TextResourceLoader


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = TextResourceLoader('tests.text')

    def test_has_package(self):
        load = TextResourceLoader('tests.text')
        self.assertTrue(hasattr(load, 'package'))

    def test_package_type(self):
        load = TextResourceLoader('tests.text')
        self.assertIsInstance(load.package, str)

    def test_package_value(self):
        load = TextResourceLoader('tests.text')
        self.assertEqual('tests.text', load.package)

    def test_package_strip(self):
        load = TextResourceLoader(' /tests.text /')
        self.assertEqual('tests.text', load.package)

    def test_has_path(self):
        load = TextResourceLoader('tests.text')
        self.assertTrue(hasattr(load, 'path'))

    def test_path_type(self):
        load = TextResourceLoader('tests.text')
        self.assertIsInstance(load.path, str)

    def test_path_value(self):
        load = TextResourceLoader('tests.text')
        self.assertEqual('resources', load.path)

    def test_has_not_found(self):
        load = TextResourceLoader('tests.text')
        self.assertTrue(hasattr(load, 'not_found'))

    def test_not_found_type(self):
        load = TextResourceLoader('tests.text')
        self.assertIsInstance(load.not_found, str)

    def test_not_found_value(self):
        load = TextResourceLoader('tests.text')
        self.assertEqual('raise', load.not_found)

    def test_has_encoding(self):
        load = TextResourceLoader('tests.text')
        self.assertTrue(hasattr(load, 'encoding'))

    def test_encoding_type(self):
        load = TextResourceLoader('tests.text')
        self.assertIsInstance(load.encoding, str)

    def test_encoding_value(self):
        load = TextResourceLoader('tests.text')
        self.assertEqual('utf-8', load.encoding)


class TestAttributes(unittest.TestCase):

    def test_base_dir_type(self):
        load = TextResourceLoader('tests.text', 'base_dir')
        self.assertIsInstance(load.path, str)

    def test_base_dir_value(self):
        load = TextResourceLoader('tests.text', 'base_dir')
        self.assertEqual('base_dir', load.path)

    def test_base_dir_strip(self):
        load = TextResourceLoader('tests.text', '/ base_dir/ ')
        self.assertEqual('base_dir', load.path)

    def test_not_found_type(self):
        load = TextResourceLoader('tests.text', 'base_dir', NotFound.WARN)
        self.assertIsInstance(load.not_found, str)

    def test_not_found_value(self):
        load = TextResourceLoader('tests.text', 'base_dir', NotFound.WARN)
        self.assertEqual('warn', load.not_found)

    def test_encoding_type(self):
        load = TextResourceLoader('tests.text', encoding='foo')
        self.assertIsInstance(load.encoding, str)

    def test_encoding_value(self):
        load = TextResourceLoader('tests.text', encoding='foo')
        self.assertEqual('foo', load.encoding)

    def test_encoding_strip(self):
        load = TextResourceLoader('tests.text', encoding=' bar ')
        self.assertEqual('bar', load.encoding)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        load = TextResourceLoader('tests.text')
        self.assertTrue(callable(load))

    @patch('swak.text.resource.pkgutil.get_data')
    def test_pkgutil_called(self, mock):
        load = TextResourceLoader('tests.text')
        _ = load('hello.txt')
        mock.assert_called_once()

    @patch('swak.text.resource.pkgutil.get_data')
    def test_pkgutil_called_correctly(self, mock):
        load = TextResourceLoader('tests.text')
        _ = load('hello.txt')
        mock.assert_called_once_with('tests.text', 'resources/hello.txt')

    def test_defaults(self):
        load = TextResourceLoader('tests.text')
        txt = load('hello.txt')
        self.assertEqual('hello\n', txt)

    def test_defaults_dir(self):
        load = TextResourceLoader('tests.text')
        txt = load('dir/world.txt')
        self.assertEqual('world\n', txt)

    def test_defaults_subdir(self):
        load = TextResourceLoader('tests.text')
        txt = load('dir/subdir/baz.txt')
        self.assertEqual('baz\n', txt)

    def test_path_file_instantiation(self):
        load = TextResourceLoader('tests.text', 'foo/bar.txt')
        txt = load()
        self.assertEqual('bar\n', txt)

    def test_path_file_call(self):
        load = TextResourceLoader('tests.text', 'foo')
        txt = load('bar.txt')
        self.assertEqual('bar\n', txt)

    def test_path_file_split(self):
        load = TextResourceLoader('tests.text', '')
        txt = load('foo/bar.txt')
        self.assertEqual('bar\n', txt)

    def test_subpath_file_instantiation(self):
        load = TextResourceLoader('tests.text', 'foo/hello/world.txt')
        txt = load()
        self.assertEqual('world\n', txt)

    def test_subpath_file_call(self):
        load = TextResourceLoader('tests.text', 'foo/hello')
        txt = load('world.txt')
        self.assertEqual('world\n', txt)

    def test_subpath_file_split(self):
        load = TextResourceLoader('tests.text', 'foo')
        txt = load('hello/world.txt')
        self.assertEqual('world\n', txt)

    def test_subpath_file_empty(self):
        load = TextResourceLoader('tests.text', '')
        txt = load('foo/hello/world.txt')
        self.assertEqual('world\n', txt)

    def test_default_raises_file_not_found(self):
        load = TextResourceLoader('tests.text')
        with self.assertRaises(FileNotFoundError):
            _ = load('non-existing')

    def test_raises_file_not_found(self):
        load = TextResourceLoader('tests.text', not_found=NotFound.RAISE)
        with self.assertRaises(FileNotFoundError):
            _ = load('non-existing')

    def test_warn_file_not_found(self):
        load = TextResourceLoader('tests.text', not_found=NotFound.WARN)
        with self.assertWarns(UserWarning):
            actual = load('non-existing')

        self.assertEqual('', actual)

    def test_ignore_file_not_found(self):
        load = TextResourceLoader('tests.text', not_found=NotFound.IGNORE)
        actual = load('non-existing')
        self.assertEqual('', actual)


class TestStrip(unittest.TestCase):

    def test_defaults_trailing(self):
        load = TextResourceLoader('tests.text')
        txt = load(' hello.txt/')
        self.assertEqual('hello\n', txt)

    def test_defaults_preceding(self):
        load = TextResourceLoader('tests.text')
        txt = load('/hello.txt ')
        self.assertEqual('hello\n', txt)

    def test_path_trailing(self):
        load = TextResourceLoader('tests.text', ' foo/')
        txt = load('bar.txt')
        self.assertEqual('bar\n', txt)

    def test_path_preceding(self):
        load = TextResourceLoader('tests.text', '/foo')
        txt = load('bar.txt')
        self.assertEqual('bar\n', txt)

    def test_subpath_trailing(self):
        load = TextResourceLoader('tests.text', ' foo/hello/')
        txt = load('world.txt')
        self.assertEqual('world\n', txt)

    def test_subpath_preceding(self):
        load = TextResourceLoader('tests.text', '/foo/hello ')
        txt = load('world.txt')
        self.assertEqual('world\n', txt)

    def test_defaults_dir_trailing(self):
        load = TextResourceLoader('tests.text')
        txt = load(' dir/world.txt/')
        self.assertEqual('world\n', txt)

    def test_defaults_dir_preceding(self):
        load = TextResourceLoader('tests.text')
        txt = load('/dir/world.txt ')
        self.assertEqual('world\n', txt)

    def test_defaults_subdir_trailing(self):
        load = TextResourceLoader('tests.text')
        txt = load(' dir/subdir/baz.txt/')
        self.assertEqual('baz\n', txt)

    def test_defaults_subdir_preceding(self):
        load = TextResourceLoader('tests.text')
        txt = load('/dir/subdir/baz.txt ')
        self.assertEqual('baz\n', txt)


class TestMisc(unittest.TestCase):

    def test_repr(self):
        load = TextResourceLoader('tests.text', path='foo')
        expected = "TextResourceLoader('tests.text', 'foo', 'raise', 'utf-8')"
        self.assertEqual(expected, repr(load))


if __name__ == '__main__':
    unittest.main()
