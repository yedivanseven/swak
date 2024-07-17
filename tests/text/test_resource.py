import unittest
from swak.text import TextResourceLoader


class TestInstantiation(unittest.TestCase):

    def test_attribute_package(self):
        load = TextResourceLoader('tests.text')
        self.assertTrue(hasattr(load, 'package'))
        self.assertEqual('tests.text', load.package)

    def test_attribute_package_strip(self):
        load = TextResourceLoader(' /tests.text /')
        self.assertEqual('tests.text', load.package)

    def test_attribute_prefix_default(self):
        load = TextResourceLoader('tests.text')
        self.assertTrue(hasattr(load, 'prefix'))

    def test_attribute_prefix(self):
        load = TextResourceLoader('tests.text', 'prefix')
        self.assertEqual('prefix', load.prefix)

    def test_attribute_prefix_strip(self):
        load = TextResourceLoader('tests.text', '/ prefix/ ')
        self.assertEqual('prefix', load.prefix)

    def test_attribute_encoding_default(self):
        load = TextResourceLoader('tests.text')
        self.assertTrue(hasattr(load, 'encoding'))

    def test_attribute_encoding(self):
        load = TextResourceLoader('tests.text', encoding='foo')
        self.assertEqual('foo', load.encoding)

    def test_attribute_encoding_strip(self):
        load = TextResourceLoader('tests.text', encoding=' bar ')
        self.assertEqual('bar', load.encoding)

    def test_callable(self):
        load = TextResourceLoader('tests.text')
        self.assertTrue(callable(load))


class TestFileLocations(unittest.TestCase):

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


class TestSlashStripping(unittest.TestCase):

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


class TestRepr(unittest.TestCase):

    def test_repr(self):
        load = TextResourceLoader('tests.text', prefix='foo')
        expected = "TextResourceLoader('tests.text', 'foo', 'utf-8')"
        self.assertEqual(expected, repr(load))


if __name__ == '__main__':
    unittest.main()
