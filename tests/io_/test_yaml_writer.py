import pickle
import unittest
from unittest.mock import patch, mock_open
from swak.io import YamlWriter, Writer, Storage, Mode


class TestInstantiation(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.yml'

    def test_is_writer(self):
        self.assertTrue(issubclass(YamlWriter, Writer))

    @patch.object(Writer, '__init__')
    def test_writer_init_called_defaults(self, init):
        _ = YamlWriter(self.path)
        init.assert_called_once_with(
            self.path,
            Storage.FILE,
            False,
            False,
            Mode.WT,
            32,
            None,
            {}
        )

    @patch.object(Writer, '__init__')
    def test_writer_init_called_custom(self, init):
        _ = YamlWriter(
            '/sone/other/file.csv',
            Storage.MEMORY,
            True,
            True,
            16,
            {'foo': 'bar'},
            {'answer': 42},
        )
        init.assert_called_once_with(
            '/sone/other/file.csv',
            Storage.MEMORY,
            True,
            True,
            Mode.WT,
            16,
            {'foo': 'bar'},
            {'answer': 42}
        )


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.yml'

    def test_has_yaml_kws(self):
        write = YamlWriter(self.path)
        self.assertTrue(hasattr(write, 'yaml_kws'))

    def test_default_yaml_kws(self):
        write = YamlWriter(self.path)
        self.assertDictEqual({}, write.yaml_kws)

    def test_custom_yaml_kws(self):
        write = YamlWriter(self.path, yaml_kws={'answer': 42})
        self.assertDictEqual({'answer': 42}, write.yaml_kws)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.yml'
        self.storage = Storage.MEMORY
        self.yml = {'hello': 'world', 'foo': {'bar': 'baz'}, 'answer': 42}

    def test_callable(self):
        write = YamlWriter(self.path)
        self.assertTrue(callable(write))

    @patch.object(Writer, '_uri_from')
    def test_uri_from_called(self, uri_from):
        uri_from.return_value = self.path
        write = YamlWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        _ = write(self.yml, 'foo', 42)
        uri_from.assert_called_once_with('foo', 42)

    @patch.object(Writer, '_managed')
    @patch.object(Writer, '_uri_from')
    def test_managed_called(self, uri_from, managed):
        uri_from.return_value = 'generated uri'
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = YamlWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        value = write(self.yml, 'foo', 42)
        managed.assert_called_once_with('generated uri')
        self.assertTupleEqual((), value)

    @patch.object(Writer, '_managed')
    @patch.object(Writer, '_uri_from')
    def test_managed_not_called(self, uri_from, managed):
        uri_from.return_value = ''
        write = YamlWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        value = write(self.yml, 'foo', 42)
        managed.assert_not_called()
        self.assertTupleEqual((), value)

    @patch('swak.io.yaml.yaml.dump')
    @patch.object(Writer, '_managed')
    def test_dump_called_defaults(self, managed, dump):
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = YamlWriter(
            self.path,
            storage=self.storage,
            overwrite=True
        )
        _ = write(self.yml)
        dump.assert_called_once_with(self.yml, mock_file.return_value)

    @patch('swak.io.yaml.yaml.dump')
    @patch.object(Writer, '_managed')
    def test_dump_called_custom(self, managed, dump):
        mock_file = mock_open()
        managed.return_value = mock_file.return_value
        write = YamlWriter(
            self.path,
            storage=self.storage,
            overwrite=True,
            yaml_kws={'answer': 42}
        )
        _ = write(self.yml)
        dump.assert_called_once_with(
            self.yml,
            mock_file.return_value,
            answer=42
        )


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.path = '/path/file.yml'

    def test_default_repr(self):
        write = YamlWriter(self.path)
        expected = ("YamlWriter('/path/file.yml', "
                    "'file', False, False, 'wt', 32.0, {}, {})")
        self.assertEqual(expected, repr(write))

    def test_custom_repr(self):
        write = YamlWriter(self.path, yaml_kws={'answer': 42})
        expected = ("YamlWriter('/path/file.yml', "
                    "'file', False, False, 'wt', 32.0, {}, {'answer': 42})")
        self.assertEqual(expected, repr(write))

    def test_pickle_works(self):
        write = YamlWriter(self.path)
        _ = pickle.loads(pickle.dumps(write))


if __name__ == '__main__':
    unittest.main()
