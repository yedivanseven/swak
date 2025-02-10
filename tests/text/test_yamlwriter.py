import pickle
import unittest
from unittest.mock import patch, mock_open
from tempfile import NamedTemporaryFile, TemporaryDirectory
from pathlib import Path
from swak.text import YamlWriter


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        with NamedTemporaryFile() as file:
            self.path = file.name
        self.write = YamlWriter(self.path)

    def test_has_path(self):
        self.assertTrue(hasattr(self.write, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.write.path)

    def test_path_like(self):
        path = Path(self.path)
        write = YamlWriter(path)
        self.assertEqual(self.path, write.path)

    def test_path_stripped(self):
        write = YamlWriter(f'  {self.path} ')
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

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.write, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.write.kwargs)

    def test_has_mode(self):
        self.assertTrue(hasattr(self.write, 'mode'))

    def test_mode(self):
        self.assertEqual('xt', self.write.mode)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.write = YamlWriter('/dir', True, True, encoding='utf-8')

    def test_overwrite(self):
        self.assertIsInstance(self.write.overwrite, bool)
        self.assertTrue(self.write.overwrite)

    def test_create(self):
        self.assertIsInstance(self.write.create, bool)
        self.assertTrue(self.write.create)

    def test_kwargs(self):
        self.assertDictEqual({'encoding': 'utf-8'}, self.write.kwargs)

    def test_mode(self):
        self.assertEqual('wt', self.write.mode)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.path = '/path/file.yml'
        self.kwargs = {'encoding': 'utf-8'}
        self.write = YamlWriter(self.path, **self.kwargs)
        self.yaml = {'Hello': 'world!'}

    def test_callable(self):
        self.assertTrue(callable(self.write))

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('tomli_w.dump')
    def test_mkdir_not_called(self, _, __, mkdir):
        _ = self.write(self.yaml)
        mkdir.assert_not_called()

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('tomli_w.dump')
    def test_mkdir_called(self, _, __, mkdir):
        write = YamlWriter(self.path, create=True)
        _ = write(self.yaml)
        mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('tomli_w.dump')
    @patch('swak.text.write.Path')
    def test_path_interpolated(self, path, *_):
        write = YamlWriter(' Hello {}, the answer is {}!  ')
        _ = write(self.yaml, 'there', '42')
        expected = 'Hello there, the answer is 42!'
        path.assert_called_once_with(expected)

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('tomli_w.dump')
    def test_open_called_no_overwrite(self, _, op, __):
        _ = self.write(self.yaml)
        op.assert_called_once_with('xt')

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('tomli_w.dump')
    def test_open_called_overwrite(self, _, op, __):
        write = YamlWriter(self.path, overwrite=True, **self.kwargs)
        _ = write(self.yaml)
        op.assert_called_once_with('wt')

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('yaml.dump')
    def test_dump_called(self, dump, op, __):
        file = op.return_value
        _ = self.write(self.yaml)
        dump.assert_called_once_with(
            self.yaml,
            file,
            **self.kwargs
        )

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('tomli_w.dump')
    def test_return_value(self, _, __, ___):
        actual = self.write(self.yaml)
        self.assertTupleEqual((), actual)

    def test_write_raises_without_create(self):
        with TemporaryDirectory() as folder:
            path = folder + '/path/does/not/exist/test.yaml'
            write = YamlWriter(path)
            with self.assertRaises(FileNotFoundError):
                _ = write(self.yaml)

    def test_write_works_with_create(self):
        with TemporaryDirectory() as folder:
            path = folder + '/path/does/not/exist/test.yaml'
            write = YamlWriter(path, create=True)
            _ = write(self.yaml)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        write = YamlWriter('/path')
        expected = "YamlWriter('/path', False, False)"
        self.assertEqual(expected, repr(write))

    def test_custom_repr(self):
        write = YamlWriter('/path', answer=42)
        expected = "YamlWriter('/path', False, False, answer=42)"
        self.assertEqual(expected, repr(write))

    def test_pickle_works(self):
        write = YamlWriter('/path')
        _ = pickle.loads(pickle.dumps(write))


if __name__ == '__main__':
    unittest.main()
