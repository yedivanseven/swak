from pathlib import Path
import unittest
from swak.text import TomlReader


def f(x: str) -> float:
    return float(x)


class TestAttributes(unittest.TestCase):

    def test_dir(self):
        t = TomlReader('/hello')
        self.assertTrue(hasattr(t, 'base_dir'))
        self.assertIsInstance(t.base_dir, str)
        self.assertEqual('/hello', t.base_dir)

    def test_dir_stripped(self):
        t = TomlReader('/ hello/ ')
        self.assertTrue(hasattr(t, 'base_dir'))
        self.assertIsInstance(t.base_dir, str)
        self.assertEqual('/hello', t.base_dir)

    def test_dir_completed(self):
        t = TomlReader('hello')
        self.assertTrue(hasattr(t, 'base_dir'))
        self.assertIsInstance(t.base_dir, str)
        self.assertEqual('/hello', t.base_dir)

    def test_default_parse_float(self):
        t = TomlReader('hello')
        self.assertTrue(hasattr(t, 'parse_float'))
        self.assertIs(t.parse_float, float)

    def test_custom_parse_float(self):
        t = TomlReader('hello', f)
        self.assertTrue(hasattr(t, 'parse_float'))
        self.assertIs(t.parse_float, f)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.dir = str(Path(__file__).parent)

    def test_callable(self):
        t = TomlReader('')
        self.assertTrue(callable(t))

    def test_read_toml(self):
        t = TomlReader(self.dir)
        actual = t('foo/bar.toml')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_path_stripped(self):
        t = TomlReader(self.dir)
        actual = t('/ foo/bar.toml/ ')
        self.assertDictEqual({'bar': {'hello': 'world'}}, actual)

    def test_path_interpolated(self):
        t = TomlReader(self.dir)
        actual = t('foo/{}/world.toml', 'hello')
        self.assertDictEqual({'world': {'answer': 42}}, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        t = TomlReader('hello')
        self.assertEqual("TomlReader('/hello', float)", repr(t))

    def test_custom_repr(self):
        t = TomlReader('hello', f)
        self.assertEqual("TomlReader('/hello', f)", repr(t))


if __name__ == '__main__':
    unittest.main()
