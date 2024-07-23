import unittest
from unittest.mock import patch
from swak.text import YamlParser
from yaml import SafeLoader, Loader


class TestAttributes(unittest.TestCase):

    def test_default_loader(self):
        y = YamlParser()
        self.assertTrue(hasattr(y, 'loader'))
        self.assertIs(y.loader, Loader)

    def test_custom_loader(self):
        y = YamlParser(SafeLoader)
        self.assertTrue(hasattr(y, 'loader'))
        self.assertIs(y.loader, SafeLoader)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.yml = """---
        foo:
          hello: world
        bar:
          answer: 42.0
        """
        self.expected = {
            'foo': {'hello': 'world'},
            'bar': {'answer': 42.0}
        }

    def test_callable(self):
        y = YamlParser()
        self.assertTrue(callable(y))

    @patch('swak.text.parse.yaml.load')
    def test_load_called(self, mock):
        y = YamlParser()
        _ = y(self.yml)
        mock.assert_called_once()

    @patch('swak.text.parse.yaml.load')
    def test_load_called_with_defaults(self, mock):
        y = YamlParser()
        _ = y(self.yml)
        mock.assert_called_once_with(self.yml, Loader)

    @patch('swak.text.parse.yaml.load')
    def test_load_called_with_custom(self, mock):
        y = YamlParser(SafeLoader)
        _ = y(self.yml)
        mock.assert_called_once_with(self.yml, SafeLoader)

    def test_parse_yaml(self):
        y = YamlParser()
        actual = y(self.yml)
        self.assertDictEqual(self.expected, actual)

    def test_parse_empty_yaml(self):
        y = YamlParser()
        actual = y('')
        self.assertDictEqual({}, actual)

    def test_parse_empty_doc_yaml(self):
        y = YamlParser()
        actual = y('---')
        self.assertDictEqual({}, actual)

    def test_parse_empty_doc_newline_yaml(self):
        y = YamlParser()
        actual = y('---\n')
        self.assertDictEqual({}, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        y = YamlParser()
        self.assertEqual("YamlParser(Loader)", repr(y))

    def test_custom_repr(self):
        y = YamlParser(SafeLoader)
        self.assertEqual("YamlParser(SafeLoader)", repr(y))


if __name__ == '__main__':
    unittest.main()
