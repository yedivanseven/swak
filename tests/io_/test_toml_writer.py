import pickle
import unittest
from unittest.mock import patch, mock_open
from swak.io import TomlWriter, Writer, Storage, Mode


class TestInstantiation(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.toml'

    def test_is_writer(self):
        self.assertTrue(issubclass(TomlWriter, Writer))

    @patch.object(Writer, '__init__')
    def test_writer_init_called_defaults(self, init):
        _ = TomlWriter(self.path)
        init.assert_called_once_with(
            self.path,
            Storage.FILE,
            False,
            False,
            Mode.WB,
            32,
            None,
            {},
            False
        )

    @patch.object(Writer, '__init__')
    def test_writer_init_called_custom(self, init):
        _ = TomlWriter(
            self.path,
            Storage.MEMORY,
            True,
            True,
            16,
            {'storage': 'kws'},
            {'toml': 'kws'},
            True
        )
        init.assert_called_once_with(
            self.path,
            Storage.MEMORY,
            True,
            True,
            Mode.WB,
            16,
            {'storage': 'kws'},
            {'toml': 'kws'},
            True
        )


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.toml'

    def test_has_prune(self):
        write = TomlWriter(self.path)
        self.assertTrue(hasattr(write, 'prune'))

    def test_default_prune(self):
        write = TomlWriter(self.path)
        self.assertIsInstance(write.prune, bool)
        self.assertFalse(write.prune)

    def test_custom_prune(self):
        write = TomlWriter(self.path, prune=True)
        self.assertTrue(write.prune)

    def test_has_toml_kws(self):
        write = TomlWriter(self.path)
        self.assertTrue(hasattr(write, 'toml_kws'))

    def test_default_toml_kws(self):
        write = TomlWriter(self.path)
        self.assertDictEqual({}, write.toml_kws)

    def test_custom_toml_kws(self):
        write = TomlWriter(self.path, toml_kws={'answer': 42})
        self.assertDictEqual({'answer': 42}, write.toml_kws)


class TestPrune(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.toml'
        self.storage = Storage.MEMORY
        self.write = TomlWriter(self.path, self.storage)
        self.toml = {
            'foo': 1,
            2: 'bar',
            'baz': None,
            'the': {'answer': 42, 'to': None, 42: 'answer', 'arr': [3, None]},
            'greet': [
                {'name': 'Hello'},
                {'name': None},
                {4: 'four'}
            ],
            'arr': [5, 6, None, 7],
        }
        self.expected = {
            'foo': 1,
            'the': {'answer': 42, 'arr': [3]},
            'greet': [
                {'name': 'Hello'},
                {},
                {}
            ],
            'arr': [5, 6, 7]
        }

    def test_prune_empty_dict(self):
        actual = self.write._pruned({})
        self.assertDictEqual({}, actual)

    def test_prune_empty_list(self):
        actual = self.write._pruned([])
        self.assertListEqual([], actual)

    def test_prune_none_key(self):
        actual = self.write._pruned({None: 'undefined'})
        self.assertDictEqual({}, actual)

    def test_prune_none_item(self):
        actual = self.write._pruned([None])
        self.assertListEqual([], actual)

    def test_prune(self):
        actual = self.write._pruned(self.toml)
        self.assertDictEqual(self.expected, actual)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.toml'
        self.storage = Storage.MEMORY
        self.toml = {
            'foo': 'bar',
            'baz': {'answer': 42},
            'greet': [
                {'name': 'Hello'},
                {'name': 'World'},
            ]
        }

    def test_callable(self):
        write = TomlWriter(self.path)
        self.assertTrue(callable(write))

    @patch.object(Writer, '_uri_from')
    def test_uri_from_called(self, uri_from):
        uri_from.return_value = self.path
        write = TomlWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        _ = write(self.toml, 'foo', 42)
        uri_from.assert_called_once_with('foo', 42)

    @patch.object(Writer, '_managed')
    @patch.object(Writer, '_uri_from')
    def test_managed_called(self, uri_from, managed):
        uri_from.return_value = 'generated uri'
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = TomlWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        value = write(self.toml, 'foo', 42)
        managed.assert_called_once_with('generated uri')
        self.assertTupleEqual((), value)

    @patch.object(Writer, '_managed')
    @patch.object(Writer, '_uri_from')
    def test_managed_not_called(self, uri_from, managed):
        uri_from.return_value = ''
        write = TomlWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        value = write(self.toml, 'foo', 42)
        managed.assert_not_called()
        self.assertTupleEqual((), value)

    @patch.object(TomlWriter, '_pruned')
    @patch.object(Writer, '_uri_from')
    def test_prune_not_called_on_skip(self, uri_from, pruned):
        uri_from.return_value = ''
        write = TomlWriter(
            self.path,
            storage=self.storage,
            overwrite=True,
            prune=True
        )
        _ = write(self.toml, 'foo', 42)
        pruned.assert_not_called()

    @patch.object(TomlWriter, '_pruned')
    def test_prune_not_called_on_prune_false(self, pruned):
        write = TomlWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        _ = write(self.toml, 'foo', 42)
        pruned.assert_not_called()

    @patch.object(TomlWriter, '_pruned')
    def test_prune_called_on_prune_true(self, pruned):
        write = TomlWriter(
            self.path,
            storage=self.storage,
            overwrite=True,
            prune=True
        )
        _ = write(self.toml, 'foo', 42)
        pruned.assert_called()

    def test_raises_if_prune_false(self):
        write = TomlWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        toml = {
            1: 'bar',
            'baz': {'answer': 42},
            'greet': [{'name': None}, {'name': 'World'}],
        }
        with self.assertRaises(TypeError):
            _ = write(toml, 'foo', 42)

    def test_works_if_prune_true(self):
        write = TomlWriter(
            self.path,
            storage=self.storage,
            overwrite=True,
            prune=True
        )
        toml = {
            1: 'bar',
            'baz': {'answer': 42},
            'greet': [{'name': None}, {'name': 'World'}],
        }
        _ = write(toml, 'foo', 42)

    @patch('swak.io.toml.tomli_w.dump')
    @patch.object(Writer, '_managed')
    def test_dump_called_defaults(self, managed, dump):
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = TomlWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        _ = write(self.toml)
        dump.assert_called_once_with(self.toml, mock_file.return_value)

    @patch('swak.io.toml.tomli_w.dump')
    @patch.object(Writer, '_managed')
    def test_dump_called_custom(self, managed, dump):
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = TomlWriter(
            self.path,
            storage=self.storage,
            overwrite=True,
            toml_kws={'answer': 42}
        )
        _ = write(self.toml)
        dump.assert_called_once_with(
            self.toml,
            mock_file.return_value,
            answer=42
        )


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.path = '/path/file.toml'

    def test_default_repr(self):
        write = TomlWriter(self.path)
        expected = ("TomlWriter('/path/file.toml', 'file', "
                    "False, False, 'wb', 32.0, {}, {}, False)")
        self.assertEqual(expected, repr(write))

    def test_custom_repr(self):
        write = TomlWriter(self.path, prune=True, toml_kws={'answer': 42})
        expected = ("TomlWriter('/path/file.toml', 'file', False, "
                    "False, 'wb', 32.0, {}, {'answer': 42}, True)")
        self.assertEqual(expected, repr(write))

    def test_pickle_works(self):
        write = TomlWriter(self.path)
        _ = pickle.loads(pickle.dumps(write))


if __name__ == '__main__':
    unittest.main()
