from pathlib import Path
import unittest
from swak.text import YamlReader
from yaml import SafeLoader, Loader


class TestAttributes(unittest.TestCase):

    def test_dir(self):
        y = YamlReader('/hello')
        self.assertTrue(hasattr(y, 'base_dir'))
        self.assertIsInstance(y.base_dir, str)
        self.assertEqual('/hello', y.base_dir)

    def test_dir_stripped(self):
        y = YamlReader('/ hello/ ')
        self.assertTrue(hasattr(y, 'base_dir'))
        self.assertIsInstance(y.base_dir, str)
        self.assertEqual('/hello', y.base_dir)

    def test_dir_completed(self):
        y = YamlReader('hello')
        self.assertTrue(hasattr(y, 'base_dir'))
        self.assertIsInstance(y.base_dir, str)
        self.assertEqual('/hello', y.base_dir)

    def test_default_loader(self):
        y = YamlReader('hello')
        self.assertTrue(hasattr(y, 'loader'))
        self.assertIs(y.loader, Loader)

    def test_custom_loader(self):
        y = YamlReader('hello', SafeLoader)
        self.assertTrue(hasattr(y, 'loader'))
        self.assertIs(y.loader, SafeLoader)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.dir = str(Path(__file__).parent)

    def test_callable(self):
        y = YamlReader('')
        self.assertTrue(callable(y))

    def test_read_yaml(self):
        y = YamlReader(self.dir)
        actual = y('foo/bar.yml')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_path_stripped(self):
        y = YamlReader(self.dir)
        actual = y('/ foo/bar.yml/ ')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_path_interpolated(self):
        y = YamlReader(self.dir)
        actual = y('foo/{}/world.yml', 'hello')
        self.assertDictEqual({'world': {'answer': 42}}, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        y = YamlReader('hello')
        self.assertEqual("YamlReader('/hello', Loader)", repr(y))

    def test_custom_repr(self):
        y = YamlReader('hello', SafeLoader)
        self.assertEqual("YamlReader('/hello', SafeLoader)", repr(y))


if __name__ == '__main__':
    unittest.main()
