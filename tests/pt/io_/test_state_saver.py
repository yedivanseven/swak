import pickle
import unittest
from unittest.mock import patch, Mock
from tempfile import TemporaryDirectory
from pathlib import Path
import torch as pt
from swak.io import Writer, Storage, Mode
from swak.pt.io import StateSaver


class TestInstantiation(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.pt'

    def test_is_writer(self):
        self.assertTrue(issubclass(StateSaver, Writer))

    @patch.object(Writer, '__init__')
    def test_writer_init_called_defaults(self, init):
        _ = StateSaver(self.path)
        init.assert_called_once_with(
            self.path,
            Storage.FILE,
            False,
            False,
            Mode.WB,
            32,
            None
        )

    @patch.object(Writer, '__init__')
    def test_writer_init_called_custom(self, init):
        _ = StateSaver(
            '/some/other/file.pt',
            Storage.MEMORY,
            True,
            True,
            16,
            {'foo': 'bar'}
        )
        init.assert_called_once_with(
            '/some/other/file.pt',
            Storage.MEMORY,
            True,
            True,
            Mode.WB,
            16,
            {'foo': 'bar'}
        )


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.storage = Storage.FILE
        self.state_dict = pt.nn.Linear(2, 3, device='cpu').state_dict()
        self.model = Mock()
        self.model.state_dict.return_value = self.state_dict
        self.dir = TemporaryDirectory()
        self.file = self.dir.name + 'file.pt'
        self.path = Path(self.file)

    def tearDown(self):
        self.dir.cleanup()

    def test_callable(self):
        write = StateSaver(self.file, storage=self.storage)
        self.assertTrue(callable(write))

    @patch.object(Writer, '_uri_from')
    def test_uri_from_called(self, uri_from):
        uri_from.return_value = self.file
        write = StateSaver(self.file, storage=self.storage)
        _ = write(self.model, 'foo', 42)
        uri_from.assert_called_once_with('foo', 42)

    @patch.object(Writer, '_managed')
    def test_managed_called(self, managed):
        with self.path.open('wb') as file:
            managed.return_value = file
            write = StateSaver(self.file, storage=self.storage, overwrite=True)
            _ = write(self.model)
        managed.assert_called_once_with(self.file)

    @patch.object(Writer, '_managed')
    @patch.object(Writer, '_uri_from')
    def test_managed_not_called(self, uri_from, managed):
        uri_from.return_value = ''
        write = StateSaver(self.file, storage=self.storage)
        _ = write(self.model)
        managed.assert_not_called()

    @patch('swak.pt.io.pt.save')
    @patch.object(Writer, '_managed')
    def test_save_called(self, managed, save):
        with self.path.open('wb') as file:
            managed.return_value = file
            write = StateSaver(self.file, storage=self.storage, overwrite=True)
            _ = write(self.model)
            save.assert_called_once_with(self.state_dict, file)

    def test_return_value(self):
        write = StateSaver(self.file, storage=self.storage)
        actual = write(self.model)
        self.assertTupleEqual((), actual)

    def test_actually_saves(self):
        write = StateSaver(self.file, storage=self.storage)
        _ = write(self.model)
        with self.path.open('rb') as file:
            actual = pt.load(file)
        pt.testing.assert_close(actual, self.state_dict)

    def test_raises_on_no_state_dict(self):
        write = StateSaver(self.file, storage=self.storage)
        with self.assertRaises(TypeError):
            _ = write(42)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.path = '/path/file.pt'

    def test_default_repr(self):
        write = StateSaver(self.path)
        expected = ("StateSaver('/path/file.pt', 'file',"
                    " False, False, 32.0, {})")
        self.assertEqual(expected, repr(write))

    def test_custom_repr(self):
        write = StateSaver(
            '/some/other/file.pt',
            Storage.MEMORY,
            True,
            True,
            16,
            {'foo': 'bar'}
        )
        expected = ("StateSaver('/some/other/file.pt', 'memory', "
                    "True, True, 16.0, {'foo': 'bar'})")
        self.assertEqual(expected, repr(write))

    def test_pickle_works(self):
        write = StateSaver(self.path)
        _ = pickle.loads(pickle.dumps(write))


if __name__ == '__main__':
    unittest.main()
