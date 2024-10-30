import pickle
import unittest
from unittest.mock import patch, mock_open
from swak.text import TomlWriter


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path'
        self.write = TomlWriter(self.path)

    def test_has_path(self):
        self.assertTrue(hasattr(self.write, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.write.path)

    def test_path_stripped(self):
        write = TomlWriter(f'  {self.path} ')
        self.assertEqual(self.path, write.path)

    def test_has_overwrite(self):
        self.assertTrue(hasattr(self.write, 'overwrite'))

    def test_overwrite(self):
        self.assertIsInstance(self.write.overwrite, bool)
        self.assertFalse(self.write.overwrite)

    def test_has_create(self):
        self.assertTrue(hasattr(self.write, 'create'))

    def test_create(self):
        self.assertIsInstance(self.write.create, bool)
        self.assertFalse(self.write.create)

    def test_has_multiline_strings(self):
        self.assertTrue(hasattr(self.write, 'multiline_strings'))

    def test_multiline_strings(self):
        self.assertIsInstance(self.write.multiline_strings, bool)
        self.assertFalse(self.write.multiline_strings)

    def test_has_indent(self):
        self.assertTrue(hasattr(self.write, 'indent'))

    def test_indent(self):
        self.assertIsInstance(self.write.indent, int)
        self.assertEqual(4, self.write.indent)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.write, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.write.kwargs)

    def test_has_mode(self):
        self.assertTrue(hasattr(self.write, 'mode'))

    def test_mode(self):
        self.assertEqual('xb', self.write.mode)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.write = TomlWriter('/path', True, True, True, 2, encoding='utf-8')

    def test_overwrite(self):
        self.assertIsInstance(self.write.overwrite, bool)
        self.assertTrue(self.write.overwrite)

    def test_create(self):
        self.assertIsInstance(self.write.create, bool)
        self.assertTrue(self.write.create)

    def test_multiline_strings(self):
        self.assertIsInstance(self.write.multiline_strings, bool)
        self.assertTrue(self.write.multiline_strings)

    def test_indent(self):
        self.assertIsInstance(self.write.indent, int)
        self.assertEqual(2, self.write.indent)

    def test_kwargs(self):
        self.assertDictEqual({'encoding': 'utf-8'}, self.write.kwargs)

    def test_mode(self):
        self.assertEqual('wb', self.write.mode)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.path = '/path/file.toml'
        self.kwargs = {'encoding': 'utf-8'}
        self.write = TomlWriter(self.path, **self.kwargs)
        self.toml = {'Hello': 'world!'}

    def test_callable(self):
        self.assertTrue(callable(self.write))

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('tomli_w.dump')
    def test_mkdir_not_called(self, _, __, mkdir):
        _ = self.write(self.toml)
        mkdir.assert_not_called()

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('tomli_w.dump')
    def test_mkdir_called(self, _, __, mkdir):
        write = TomlWriter(self.path, create=True)
        _ = write(self.toml)
        mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('tomli_w.dump')
    @patch('swak.text.write.Path')
    def test_path_interpolated(self, path, *_):
        write = TomlWriter(' Hello {}, the answer is {}!  ')
        _ = write(self.toml, 'there', '42')
        expected = 'Hello there, the answer is 42!'
        path.assert_called_once_with(expected)

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('tomli_w.dump')
    def test_open_called_no_overwrite(self, _, op, __):
        _ = self.write(self.toml)
        op.assert_called_once_with('xb', **self.kwargs)

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('tomli_w.dump')
    def test_open_called_overwrite(self, _, op, __):
        write = TomlWriter(self.path, overwrite=True, **self.kwargs)
        _ = write(self.toml)
        op.assert_called_once_with('wb', **self.kwargs)

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('tomli_w.dump')
    def test_dump_called(self, dump, op, __):
        file = op.return_value
        _ = self.write(self.toml)
        dump.assert_called_once_with(
            self.toml,
            file,
            multiline_strings=False,
            indent=4
        )

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('tomli_w.dump')
    def test_return_value(self, _, __, ___):
        actual = self.write(self.toml)
        self.assertTupleEqual((), actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        write = TomlWriter('/path')
        expected = "TomlWriter('/path', False, False, False, 4)"
        self.assertEqual(expected, repr(write))

    def test_custom_repr(self):
        write = TomlWriter('/path', answer=42)
        expected = "TomlWriter('/path', False, False, False, 4, answer=42)"
        self.assertEqual(expected, repr(write))

    def test_pickle_works(self):
        write = TomlWriter('/path')
        _ = pickle.dumps(write)


if __name__ == '__main__':
    unittest.main()
