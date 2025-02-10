import pickle
import unittest
from unittest.mock import patch, mock_open
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from swak.text import JsonWriter


def f(x):
    return repr(x)


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path'
        self.write = JsonWriter(self.path)

    def test_has_path(self):
        self.assertTrue(hasattr(self.write, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.write.path)

    def test_path_like(self):
        write = JsonWriter(Path(self.path))
        self.assertEqual(self.path, write.path)

    def test_path_stripped(self):
        write = JsonWriter(f'  {self.path} ')
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

    def test_has_gzipped(self):
        self.assertTrue(hasattr(self.write, 'gzipped'))

    def test_gzipped(self):
        self.assertIsNone(self.write.gzipped)

    def test_has_prune(self):
        self.assertTrue(hasattr(self.write, 'prune'))

    def test_prune(self):
        self.assertIsInstance(self.write.prune, bool)
        self.assertFalse(self.write.prune)

    def test_has_default(self):
        self.assertTrue(hasattr(self.write, 'default'))

    def test_multiline_strings(self):
        self.assertIsNone(self.write.default)

    def test_has_indent(self):
        self.assertTrue(hasattr(self.write, 'indent'))

    def test_indent(self):
        self.assertIsNone(self.write.indent)

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
        self.write = JsonWriter(
            '/path',
            True,
            True,
            True,
            True,
            f,
            4,
            encoding='utf-8'
        )

    def test_overwrite(self):
        self.assertIsInstance(self.write.overwrite, bool)
        self.assertTrue(self.write.overwrite)

    def test_create(self):
        self.assertIsInstance(self.write.create, bool)
        self.assertTrue(self.write.create)

    def test_gzipped(self):
        self.assertIsInstance(self.write.gzipped, bool)
        self.assertTrue(self.write.prune)

    def test_prune(self):
        self.assertIsInstance(self.write.prune, bool)
        self.assertTrue(self.write.prune)

    def test_default(self):
        self.assertIs(self.write.default, f)

    def test_indent(self):
        self.assertIsInstance(self.write.indent, int)
        self.assertEqual(4, self.write.indent)

    def test_kwargs(self):
        self.assertDictEqual({'encoding': 'utf-8'}, self.write.kwargs)

    def test_mode(self):
        self.assertEqual('wt', self.write.mode)

    def test_mode_kwarg_purged(self):
        write = JsonWriter(mode='+w')
        self.assertDictEqual({}, write.kwargs)


class TestPlainUsage(unittest.TestCase):

    def setUp(self):
        with NamedTemporaryFile() as file:
            self.path = file.name
        self.kwargs = {'encoding': 'utf-8'}
        self.write = JsonWriter(self.path, **self.kwargs)
        self.json = {'Hello': 'world!'}

    def test_callable(self):
        self.assertTrue(callable(self.write))

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('json.dump')
    def test_mkdir_not_called(self, _, __, mkdir):
        _ = self.write(self.json)
        mkdir.assert_not_called()

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('json.dump')
    def test_mkdir_called(self, _, __, mkdir):
        write = JsonWriter(self.path, create=True)
        _ = write(self.json)
        mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('swak.text.write.Path')
    def test_path_interpolated(self, path, *_):
        write = JsonWriter(' Hello {}, the answer is {}!  ')
        _ = write(self.json, 'there', '42')
        expected = 'Hello there, the answer is 42!'
        path.assert_called_once_with(expected)

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('json.dump')
    def test_open_called_no_overwrite(self, _, op, __):
        _ = self.write(self.json)
        op.assert_called_once_with('xt', **self.kwargs)

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('json.dump')
    def test_open_called_overwrite(self, _, op, __):
        write = JsonWriter(self.path, overwrite=True, **self.kwargs)
        _ = write(self.json)
        op.assert_called_once_with('wt', **self.kwargs)

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('json.dump')
    def test_open_called_with_mode_purged(self, _, op, __):
        write = JsonWriter(mode='+w')
        _ = write(self.json)
        op.assert_called_once_with('xt')

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('json.dump')
    def test_dump_called_default(self, dump, op, __):
        file = op.return_value
        _ = self.write(self.json)
        dump.assert_called_once_with(
            self.json,
            file,
            skipkeys=False,
            ensure_ascii=False,
            indent=None,
            default=None
        )

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('json.dump')
    def test_dump_called_custom(self, dump, op, __):
        write = JsonWriter(self.path, prune=True, default=f, indent=4)
        file = op.return_value
        _ = write(self.json)
        dump.assert_called_once_with(
            self.json,
            file,
            skipkeys=True,
            ensure_ascii=False,
            indent=4,
            default=f
        )

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('json.dump')
    def test_return_value(self, _, __, ___):
        actual = self.write(self.json)
        self.assertTupleEqual((), actual)

    def test_write_raises_without_create(self):
        with TemporaryDirectory() as folder:
            path = folder + '/path/does/not/exist/test.json'
            write = JsonWriter(path)
            with self.assertRaises(FileNotFoundError):
                _ = write(self.json)

    def test_write_works_with_create(self):
        with TemporaryDirectory() as folder:
            path = folder + '/path/does/not/exist/test.json'
            write = JsonWriter(path, create=True)
            _ = write(self.json)


class TestGzipUsage(unittest.TestCase):

    def setUp(self):
        with NamedTemporaryFile() as file:
            self.file = file.name
        self.zipfile = self.file + '.gz'
        self.kwargs = {'encoding': 'utf-8'}
        self.implicit = JsonWriter(self.zipfile, **self.kwargs)
        self.explicit = JsonWriter(self.file, gzipped=True, **self.kwargs)
        self.json = {'Hello': 'world!'}

    @patch('pathlib.Path.mkdir')
    @patch('gzip.open', new_callable=mock_open)
    @patch('json.dump')
    def test_implicit_open_called_no_overwrite(self, _, op, __):
        _ = self.implicit(self.json)
        op.assert_called_once_with(Path(self.zipfile), 'xt', encoding='utf-8')

    @patch('pathlib.Path.mkdir')
    @patch('gzip.open', new_callable=mock_open)
    @patch('json.dump')
    def test_explicit_open_called_no_overwrite(self, _, op, __):
        _ = self.explicit(self.json)
        op.assert_called_once_with(Path(self.file), 'xt', encoding='utf-8')

    @patch('pathlib.Path.mkdir')
    @patch('gzip.open', new_callable=mock_open)
    @patch('json.dump')
    def test_implicit_open_called_overwrite(self, _, op, __):
        write = JsonWriter(self.zipfile, overwrite=True, **self.kwargs)
        _ = write(self.json)
        op.assert_called_once_with(Path(self.zipfile), 'wt', encoding='utf-8')

    @patch('pathlib.Path.mkdir')
    @patch('gzip.open', new_callable=mock_open)
    @patch('json.dump')
    def test_explicit_open_called_overwrite(self, _, op, __):
        write = JsonWriter(
            self.file,
            gzipped=True,
            overwrite=True,
            **self.kwargs
        )
        _ = write(self.json)
        op.assert_called_once_with(Path(self.file), 'wt', encoding='utf-8')

    @patch('pathlib.Path.mkdir')
    @patch('gzip.open', new_callable=mock_open)
    @patch('json.dump')
    def test_implicit_open_called_with_mode_purged(self, _, op, __):
        write = JsonWriter(self.zipfile, mode='+w')
        _ = write(self.json)
        op.assert_called_once_with(Path(self.zipfile), 'xt')

    @patch('pathlib.Path.mkdir')
    @patch('gzip.open', new_callable=mock_open)
    @patch('json.dump')
    def test_explicit_open_called_with_mode_purged(self, _, op, __):
        write = JsonWriter(self.file, gzipped=True, mode='+w')
        _ = write(self.json)
        op.assert_called_once_with(Path(self.file), 'xt')

    @patch('pathlib.Path.mkdir')
    @patch('gzip.open', new_callable=mock_open)
    @patch('json.dump')
    def test_implicit_dump_called_default(self, dump, op, __):
        file = op.return_value
        _ = self.implicit(self.json)
        dump.assert_called_once_with(
            self.json,
            file,
            skipkeys=False,
            ensure_ascii=False,
            indent=None,
            default=None
        )

    @patch('pathlib.Path.mkdir')
    @patch('gzip.open', new_callable=mock_open)
    @patch('json.dump')
    def test_explicit_dump_called_default(self, dump, op, __):
        file = op.return_value
        _ = self.explicit(self.json)
        dump.assert_called_once_with(
            self.json,
            file,
            skipkeys=False,
            ensure_ascii=False,
            indent=None,
            default=None
        )

    @patch('pathlib.Path.mkdir')
    @patch('gzip.open', new_callable=mock_open)
    @patch('json.dump')
    def test_implicit_dump_called_custom(self, dump, op, __):
        write = JsonWriter(self.zipfile, prune=True, default=f, indent=4)
        file = op.return_value
        _ = write(self.json)
        dump.assert_called_once_with(
            self.json,
            file,
            skipkeys=True,
            ensure_ascii=False,
            indent=4,
            default=f
        )

    @patch('pathlib.Path.mkdir')
    @patch('gzip.open', new_callable=mock_open)
    @patch('json.dump')
    def test_explicit_dump_called_custom(self, dump, op, __):
        write = JsonWriter(
            self.file,
            gzipped=True,
            prune=True,
            default=f,
            indent=4
        )
        file = op.return_value
        _ = write(self.json)
        dump.assert_called_once_with(
            self.json,
            file,
            skipkeys=True,
            ensure_ascii=False,
            indent=4,
            default=f
        )

    @patch('pathlib.Path.mkdir')
    @patch('gzip.open', new_callable=mock_open)
    @patch('json.dump')
    def test_implicit_return_value(self, _, __, ___):
        actual = self.implicit(self.json)
        self.assertTupleEqual((), actual)

    @patch('pathlib.Path.mkdir')
    @patch('gzip.open', new_callable=mock_open)
    @patch('json.dump')
    def test_explicit_return_value(self, _, __, ___):
        actual = self.explicit(self.json)
        self.assertTupleEqual((), actual)

    def test_implicit_write_raises_without_create(self):
        with TemporaryDirectory() as folder:
            path = folder + '/path/does/not/exist/test.json.gz'
            write = JsonWriter(path)
            with self.assertRaises(FileNotFoundError):
                _ = write(self.json)

    def test_explicit_write_raises_without_create(self):
        with TemporaryDirectory() as folder:
            path = folder + '/path/does/not/exist/test.json'
            write = JsonWriter(path, gzipped=True)
            with self.assertRaises(FileNotFoundError):
                _ = write(self.json)

    def test_implicit_write_works_with_create(self):
        with TemporaryDirectory() as folder:
            path = folder + '/path/does/not/exist/test.json.gz'
            write = JsonWriter(path, create=True)
            _ = write(self.json)

    def test_explicit_write_works_with_create(self):
        with TemporaryDirectory() as folder:
            path = folder + '/path/does/not/exist/test.json'
            write = JsonWriter(path, gzipped=True, create=True)
            _ = write(self.json)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        write = JsonWriter('/path')
        expected = "JsonWriter('/path', False, False, None, False, None, None)"
        self.assertEqual(expected, repr(write))

    def test_custom_repr(self):
        write = JsonWriter('/path', True, prune=True, hello=42)
        expected = ("JsonWriter('/path', True, False, "
                    "None, True, None, None, hello=42)")
        self.assertEqual(expected, repr(write))

    def test_pickle_works(self):
        write = JsonWriter('/path')
        _ = pickle.loads(pickle.dumps(write))


if __name__ == '__main__':
    unittest.main()
