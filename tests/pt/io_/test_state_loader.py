import pickle
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import torch as pt
from swak.io import Storage, Reader, Mode, NotFound
from swak.pt.io import StateLoader


class TestInstantiation(unittest.TestCase):

    def test_is_reader(self):
        self.assertTrue(issubclass(StateLoader, Reader))

    @patch.object(Reader, '__init__')
    def test_reader_init_called_defaults(self, init):
        _ = StateLoader('/path/to/file.pt')
        init.assert_called_once_with(
            '/path/to/file.pt',
            Storage.FILE,
            Mode.RB,
            32,
            None,
            None,
            True,
            'raise'
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_custom(self, init):
        _ = StateLoader(
            '/path/to/other/file.pt',
            Storage.MEMORY,
            16,
            {'storage': 'kws'},
            'cpu',
            False,
            'ignore'
        )
        init.assert_called_once_with(
            '/path/to/other/file.pt',
            Storage.MEMORY,
            Mode.RB,
            16,
            {'storage': 'kws'},
            'cpu',
            False,
            'ignore'
        )


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.path = '/path/to/file.pt'

    def test_has_map_location(self):
        read = StateLoader(self.path)
        self.assertTrue(hasattr(read, 'map_location'))

    def test_default_map_location(self):
        read = StateLoader(self.path)
        self.assertIsNone(read.map_location)

    def test_custom_map_location(self):
        read = StateLoader(self.path, map_location='cpu')
        self.assertEqual('cpu', read.map_location)

    def test_has_merge(self):
        read = StateLoader(self.path)
        self.assertTrue(hasattr(read, 'merge'))

    def test_default_merge(self):
        read = StateLoader(self.path)
        self.assertIsInstance(read.merge, bool)
        self.assertTrue(read.merge)

    def test_custom_merge(self):
        read = StateLoader(self.path, merge=False)
        self.assertIsInstance(read.merge, bool)
        self.assertFalse(read.merge)

    def test_has_not_found(self):
        read = StateLoader(self.path)
        self.assertTrue(hasattr(read, 'not_found'))

    def test_default_not_found(self):
        read = StateLoader(self.path)
        self.assertEqual('raise', read.not_found)

    def test_custom_not_found_str(self):
        read = StateLoader(self.path, not_found='ignore')
        self.assertEqual('ignore', read.not_found)

    def test_custom_not_found_enum(self):
        read = StateLoader(self.path, not_found=NotFound.IGNORE)
        self.assertEqual('ignore', read.not_found)

    def test_wrong_not_found_raises(self):
        with self.assertRaises(ValueError):
            _ = StateLoader(self.path, not_found='wrong')


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.storage = Storage.FILE
        self.model = pt.nn.Linear(2, 3, device='cpu', bias=False)
        self.dir = TemporaryDirectory()
        self.file = self.dir.name + '/file.pt'
        self.path = Path(self.file)
        pt.save(self.model.state_dict(), self.file)

    def tearDown(self):
        self.dir.cleanup()

    def test_callable(self):
        read = StateLoader(self.file, self.storage)
        self.assertTrue(callable(read))

    @patch.object(Reader, '_non_root')
    def test_non_root_called_default(self, non_root):
        read = StateLoader(self.file, self.storage)
        non_root.return_value = self.file
        _ = read(self.model)
        non_root.assert_called_once_with(self.file)

    @patch.object(Reader, '_non_root')
    def test_non_root_called_custom(self, non_root):
        read = StateLoader('/some/other/path.pt', self.storage)
        non_root.return_value = self.file
        _ = read(self.model)
        non_root.assert_called_once_with('/some/other/path.pt')

    @patch.object(Reader, '_non_root')
    def test_non_root_called_interpolated(self, non_root):
        read = StateLoader('/some/{}/path.pt', self.storage)
        non_root.return_value = self.file
        _ = read(self.model, 'other')
        non_root.assert_called_once_with('/some/other/path.pt')

    @patch.object(Reader, '_managed')
    def test_managed_called(self, managed):
        read = StateLoader(self.file, self.storage)
        with self.path.open('rb') as file:
            managed.return_value = file
            _ = read(self.model)
        managed.assert_called_once_with(self.file)

    @patch.object(Reader, '_managed')
    @patch('swak.pt.io.pt.load')
    def test_pt_load_called_defaults(self, load, managed):
        read = StateLoader(self.file, self.storage)
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.model.state_dict()
            _ = read(self.model)
            load.assert_called_once_with(file, None, weights_only=True)

    @patch.object(Reader, '_managed')
    @patch('swak.pt.io.pt.load')
    def test_pt_load_called_custom(self, load, managed):
        read = StateLoader(self.file, self.storage, map_location='cpu')
        with self.path.open('rb') as file:
            managed.return_value = file
            load.return_value = self.model.state_dict()
            _ = read(self.model)
            load.assert_called_once_with(file, 'cpu', weights_only=True)

    def test_default_raises_on_file_not_found(self):
        read = StateLoader('/some/other/file.pt', self.storage)
        with self.assertRaises(FileNotFoundError):
            _ = read(self.model)

    def test_custom_raises_on_file_not_found_merge_true(self):
        read = StateLoader(
            '/some/other/file.pt',
            self.storage,
            not_found='raise',
            merge=True
        )
        with self.assertRaises(FileNotFoundError):
            _ = read(self.model)

    def test_custom_raises_on_file_not_found_merge_false(self):
        read = StateLoader(
            '/some/other/file.pt',
            self.storage,
            not_found='raise',
            merge=False
        )
        with self.assertRaises(FileNotFoundError):
            _ = read(self.model)

    def test_warns_on_file_not_found_merge_true(self):
        read = StateLoader(
            '/some/other/file.pt',
            self.storage,
            not_found='warn',
            merge=True
        )
        with self.assertWarns(UserWarning):
            actual = read(self.model)
        self.assertIs(actual, self.model)
        pt.testing.assert_close(actual.state_dict(), self.model.state_dict())

    def test_warns_on_file_not_found_merge_false(self):
        read = StateLoader(
            '/some/other/file.pt',
            self.storage,
            not_found='warn',
            merge=False
        )
        with self.assertWarns(UserWarning):
            actual = read(self.model)
        self.assertIs(actual, self.model)
        pt.testing.assert_close(actual.state_dict(), self.model.state_dict())

    def test_ignores_file_not_found_merge_true(self):
        read = StateLoader(
            '/some/other/file.pt',
            self.storage,
            not_found=NotFound.IGNORE,
            merge=True
        )
        actual = read(self.model)
        self.assertIs(actual, self.model)
        pt.testing.assert_close(actual.state_dict(), self.model.state_dict())

    def test_ignores_file_not_found_merge_false(self):
        read = StateLoader(
            '/some/other/file.pt',
            self.storage,
            not_found=NotFound.IGNORE,
            merge=False
        )
        actual = read(self.model)
        self.assertIs(actual, self.model)
        pt.testing.assert_close(actual.state_dict(), self.model.state_dict())

    def test_reads_fewer_keys_merge_true(self):
        model = pt.nn.Linear(2, 3, device='cpu')
        bias = model.state_dict()['bias'].detach().clone()
        read = StateLoader(self.file, self.storage, merge=True)
        actual = read(model)
        self.assertIs(actual, model)
        pt.testing.assert_close(
            actual.state_dict()['weight'],
            self.model.state_dict()['weight']
        )
        pt.testing.assert_close(
            actual.state_dict()['bias'],
            bias
        )

    def test_raises_on_fewer_keys_merge_false(self):
        model = pt.nn.Linear(2, 3, device='cpu')
        read = StateLoader(self.file, self.storage, merge=False)
        with self.assertRaises(RuntimeError):
            _ = read(model)

    def test_reads_same_keys_merge_true(self):
        checkpoint = pt.nn.Linear(2, 3, device='cpu')
        pt.save(checkpoint.state_dict(), self.file)
        model = pt.nn.Linear(2, 3, device='cpu')
        read = StateLoader(self.file, self.storage, merge=True)
        actual = read(model)
        self.assertIs(actual, model)
        pt.testing.assert_close(actual.state_dict(), checkpoint.state_dict())

    def test_reads_same_keys_merge_false(self):
        checkpoint = pt.nn.Linear(2, 3, device='cpu')
        pt.save(checkpoint.state_dict(), self.file)
        model = pt.nn.Linear(2, 3, device='cpu')
        read = StateLoader(self.file, self.storage, merge=False)
        actual = read(model)
        self.assertIs(actual, model)
        pt.testing.assert_close(actual.state_dict(), checkpoint.state_dict())

    def test_raises_on_more_keys_merge_true(self):
        checkpoint = pt.nn.Linear(2, 3, device='cpu')
        pt.save(checkpoint.state_dict(), self.file)
        model = pt.nn.Linear(2, 3, device='cpu', bias=False)
        read = StateLoader(self.file, self.storage, merge=True)
        with self.assertRaises(RuntimeError):
            _ = read(model)

    def test_raises_on_more_keys_merge_false(self):
        checkpoint = pt.nn.Linear(2, 3, device='cpu')
        pt.save(checkpoint.state_dict(), self.file)
        model = pt.nn.Linear(2, 3, device='cpu', bias=False)
        read = StateLoader(self.file, self.storage, merge=False)
        with self.assertRaises(RuntimeError):
            _ = read(model)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        read = StateLoader('/path/file.pt')
        expected = ("StateLoader('/path/file.pt', 'file', "
                    "32.0, {}, None, True, 'raise')")
        self.assertEqual(expected, repr(read))

    def test_custom_repr(self):
        read = StateLoader(
                '/path/file.pt',
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                'cpu',
                False,
            'ignore'
        )
        expected = ("StateLoader('/path/file.pt', 'memory', 16.0, "
                    "{'storage': 'kws'}, 'cpu', False, 'ignore')")
        self.assertEqual(expected, repr(read))

    def test_pickle_works(self):
        read = StateLoader('/path/file.pt')
        _ = pickle.loads(pickle.dumps(read))


if __name__ == '__main__':
    unittest.main()
