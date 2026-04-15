import pickle
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import torch as pt
from swak.io import Storage, Reader, Mode
from swak.pt.io import ModelLoader


class TestInstantiation(unittest.TestCase):

    def test_is_reader(self):
        self.assertTrue(issubclass(ModelLoader, Reader))

    @patch.object(Reader, '__init__')
    def test_reader_init_called_defaults(self, init):
        _ = ModelLoader()
        init.assert_called_once_with(
            '', Storage.FILE, Mode.RB, 32, None, None
        )

    @patch.object(Reader, '__init__')
    def test_reader_init_called_custom(self, init):
        _ = ModelLoader(
            '/path/to/file.pt',
            Storage.MEMORY,
            16,
            {'storage': 'kws'},
            'cpu'
        )
        init.assert_called_once_with(
            '/path/to/file.pt',
            Storage.MEMORY,
            Mode.RB,
            16,
            {'storage': 'kws'},
            'cpu'
        )


class TestAttributes(unittest.TestCase):

    def test_has_map_location(self):
        read = ModelLoader()
        self.assertTrue(hasattr(read, 'map_location'))

    def test_default_map_location(self):
        read = ModelLoader()
        self.assertIsNone(read.map_location)

    def test_custom_map_location(self):
        read = ModelLoader(map_location='cpu')
        self.assertEqual('cpu', read.map_location)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.storage = Storage.FILE
        self.model = pt.nn.Linear(2, 3, device='cpu')
        self.dir = TemporaryDirectory()
        self.file = self.dir.name + '/file.pt'
        self.path = Path(self.file)
        pt.save(self.model, self.file)

    def tearDown(self):
        self.dir.cleanup()

    def test_callable(self):
        read = ModelLoader()
        self.assertTrue(callable(read))

    @patch.object(Reader, '_non_root')
    def test_non_root_called_default(self, non_root):
        read = ModelLoader(self.file, self.storage)
        non_root.return_value = self.file
        _ = read()
        non_root.assert_called_once_with('')

    @patch.object(Reader, '_non_root')
    def test_non_root_called_custom(self, non_root):
        read = ModelLoader('/some/other/path.pt', self.storage)
        non_root.return_value = self.file
        _ = read(self.file)
        non_root.assert_called_once_with(self.file)

    @patch.object(Reader, '_non_root')
    @patch.object(Reader, '_managed')
    def test_managed_called(self, managed, non_root):
        read = ModelLoader(self.file, self.storage)
        non_root.return_value = self.file
        with self.path.open('rb') as file:
            managed.return_value = file
            _ = read()
        managed.assert_called_once_with(self.file)

    @patch.object(Reader, '_managed')
    @patch('swak.pt.io.pt.load')
    def test_pt_load_called_defaults(self, load, managed):
        read = ModelLoader(self.file, self.storage)
        with self.path.open('rb') as file:
            managed.return_value = file
            _ = read()
            load.assert_called_once_with(file, None, weights_only=False)

    @patch.object(Reader, '_managed')
    @patch('swak.pt.io.pt.load')
    def test_pt_load_called_custom(self, load, managed):
        read = ModelLoader(self.file, self.storage, map_location='cpu')
        with self.path.open('rb') as file:
            managed.return_value = file
            _ = read()
            load.assert_called_once_with(file, 'cpu', weights_only=False)

    def test_return_value(self):
        read = ModelLoader(self.file, self.storage)
        actual = read()
        self.assertIsInstance(actual, pt.nn.Linear)
        for p1, p2 in zip(self.model.parameters(), actual.parameters()):
            self.assertTrue(pt.allclose(p1, p2))


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        read = ModelLoader()
        expected = "ModelLoader('/', 'file', 32.0, {}, None)"
        self.assertEqual(expected, repr(read))

    def test_custom_repr(self):
        read = ModelLoader(
                '/path/file.pt',
                Storage.MEMORY,
                16,
                {'storage': 'kws'},
                'cpu'
        )
        expected = ("ModelLoader('/path/file.pt', 'memory', "
                    "16.0, {'storage': 'kws'}, 'cpu')")
        self.assertEqual(expected, repr(read))

    def test_pickle_works(self):
        read = ModelLoader()
        _ = pickle.loads(pickle.dumps(read))


if __name__ == '__main__':
    unittest.main()
