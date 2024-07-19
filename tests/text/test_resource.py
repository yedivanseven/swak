import unittest
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

    def test_has_prefix(self):
        load = TextResourceLoader('tests.text')
        self.assertTrue(hasattr(load, 'base_dir'))

    def test_prefix_type(self):
        load = TextResourceLoader('tests.text')
        self.assertIsInstance(load.base_dir, str)

    def test_prefix_value(self):
        load = TextResourceLoader('tests.text')
        self.assertEqual('resources', load.base_dir)

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

    def test_prefix_type(self):
        load = TextResourceLoader('tests.text', 'base_dir')
        self.assertIsInstance(load.base_dir, str)

    def test_prefix_value(self):
        load = TextResourceLoader('tests.text', 'base_dir')
        self.assertEqual('base_dir', load.base_dir)

    def test_prefix_strip(self):
        load = TextResourceLoader('tests.text', '/ base_dir/ ')
        self.assertEqual('base_dir', load.base_dir)

    def test_encoding_type(self):
        load = TextResourceLoader('tests.text', encoding='foo')
        self.assertIsInstance(load.encoding, str)

    def test_encoding_value(self):
        load = TextResourceLoader('tests.text', encoding='foo')
        self.assertEqual('foo', load.encoding)

    def test_encoding_strip(self):
        load = TextResourceLoader('tests.text', encoding=' bar ')
        self.assertEqual('bar', load.encoding)

    def test_callable(self):
        load = TextResourceLoader('tests.text')
        self.assertTrue(callable(load))


class TestUsage(unittest.TestCase):

    def test_defaults(self):
        load = TextResourceLoader('tests.text')
        txt = load('hello.txt')
        self.assertEqual('hello\n', txt)

    def test_prefix_file(self):
        load = TextResourceLoader('tests.text', 'foo')
        txt = load('bar.txt')
        self.assertEqual('bar\n', txt)

    def test_subprefix_file(self):
        load = TextResourceLoader('tests.text', 'foo/hello')
        txt = load('world.txt')
        self.assertEqual('world\n', txt)

    def test_defaults_dir(self):
        load = TextResourceLoader('tests.text')
        txt = load('dir/world.txt')
        self.assertEqual('world\n', txt)

    def test_defaults_subdir(self):
        load = TextResourceLoader('tests.text')
        txt = load('dir/subdir/baz.txt')
        self.assertEqual('baz\n', txt)


class TestInterpolation(unittest.TestCase):

    def test_prefix_only(self):
        load = TextResourceLoader('tests.text', 'resources/{}')
        txt = load('subdir/baz.txt', 'dir')
        self.assertEqual('baz\n', txt)

    def test_path_only(self):
        load = TextResourceLoader('tests.text', )
        txt = load('{}/{}/baz.txt', 'dir', 'subdir')
        self.assertEqual('baz\n', txt)

    def test_prefix_and_path(self):
        load = TextResourceLoader('tests.text', 'resources/{}')
        txt = load('{}/baz.txt', 'dir', 'subdir')
        self.assertEqual('baz\n', txt)


class TestStrip(unittest.TestCase):

    def test_defaults_trailing(self):
        load = TextResourceLoader('tests.text')
        txt = load(' hello.txt/')
        self.assertEqual('hello\n', txt)

    def test_defaults_preceding(self):
        load = TextResourceLoader('tests.text')
        txt = load('/hello.txt ')
        self.assertEqual('hello\n', txt)

    def test_prefix_trailing(self):
        load = TextResourceLoader('tests.text', ' foo/')
        txt = load('bar.txt')
        self.assertEqual('bar\n', txt)

    def test_prefix_preceding(self):
        load = TextResourceLoader('tests.text', '/foo')
        txt = load('bar.txt')
        self.assertEqual('bar\n', txt)

    def test_subprefix_trailing(self):
        load = TextResourceLoader('tests.text', ' foo/hello/')
        txt = load('world.txt')
        self.assertEqual('world\n', txt)

    def test_subprefix_preceding(self):
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
        load = TextResourceLoader('tests.text', base_dir='foo')
        expected = "TextResourceLoader('tests.text', 'foo', 'utf-8')"
        self.assertEqual(expected, repr(load))


if __name__ == '__main__':
    unittest.main()
