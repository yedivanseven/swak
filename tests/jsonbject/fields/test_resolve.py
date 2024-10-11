import unittest
from unittest.mock import patch
from swak.jsonobject.fields import resolve


class TestResolve(unittest.TestCase):

    @patch('swak.jsonobject.fields.resolve.Path')
    def test_path_called(self, mock):
        _ = resolve('path')
        mock.assert_called_once_with('path')

    @patch('swak.jsonobject.fields.resolve.Path.resolve')
    def test_resolve_called(self, mock):
        _ = resolve('path')
        mock.assert_called_once_with()

    @patch('swak.jsonobject.fields.resolve.Path.resolve', return_value=42)
    def test_return_value_cast(self, mock):
        actual = resolve('path')
        self.assertEqual('42', actual)


if __name__ == '__main__':
    unittest.main()
