import unittest
from unittest.mock import patch
from swak.text import TomlParser


def f(x: str) -> float:
    return float(x) + 1.0


class TestAttributes(unittest.TestCase):

    def test_default_parse_float(self):
        t = TomlParser()
        self.assertTrue(hasattr(t, 'parse_float'))
        self.assertIs(t.parse_float, float)

    def test_custom_parse_float(self):
        t = TomlParser(f)
        self.assertTrue(hasattr(t, 'parse_float'))
        self.assertIs(t.parse_float, f)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.toml = """
        [foo]
        hello = "world"
        
        [bar]
        answer = 42.0
        """
        self.expected = {
            'foo': {'hello': 'world'},
            'bar': {'answer': 42.0}
        }

    def test_callable(self):
        t = TomlParser()
        self.assertTrue(callable(t))

    @patch('swak.text.parse.tomllib.loads')
    def test_loads_called(self, mock):
        t = TomlParser()
        _ = t(self.toml)
        mock.assert_called_once()

    @patch('swak.text.parse.tomllib.loads')
    def test_loads_called_with_default(self, mock):
        t = TomlParser()
        _ = t(self.toml)
        mock.assert_called_once_with(self.toml, parse_float=float)

    @patch('swak.text.parse.tomllib.loads')
    def test_loads_called_with_custom(self, mock):
        t = TomlParser(f)
        _ = t(self.toml)
        mock.assert_called_once_with(self.toml, parse_float=f)

    def test_read_toml(self):
        t = TomlParser()
        actual = t(self.toml)
        self.assertDictEqual(self.expected, actual)

    def test_default_float_parser(self):
        t = TomlParser()
        actual = t(self.toml)
        self.assertIsInstance(actual['bar']['answer'], float)
        self.assertEqual(42.0, actual['bar']['answer'])

    def test_custom_float_parser(self):
        t = TomlParser(f)
        actual = t(self.toml)
        self.assertIsInstance(actual['bar']['answer'], float)
        self.assertEqual(43.0, actual['bar']['answer'])


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        t = TomlParser()
        self.assertEqual("TomlParser(float)", repr(t))

    def test_custom_repr(self):
        t = TomlParser(f)
        self.assertEqual("TomlParser(f)", repr(t))


if __name__ == '__main__':
    unittest.main()
