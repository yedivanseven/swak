import pickle
import unittest
from unittest.mock import patch, Mock, mock_open
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.implementations.local import LocalFileSystem
import torch as pt
from swak.pt.train import Checkpoint
from swak.io import Storage


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.check = Checkpoint()

    def test_has_path(self):
        self.assertTrue(hasattr(self.check, 'path'))

    def test_path(self):
        self.assertEqual('/tmp/checkpoint.pt', self.check.path)

    def test_has_storage(self):
        self.assertTrue(hasattr(self.check, 'storage'))

    def test_storage(self):
        self.assertIsInstance(self.check.storage, str)
        self.assertEqual('memory', self.check.storage)

    def test_has_chunk_size(self):
        self.assertTrue(hasattr(self.check, 'chunk_size'))

    def test_chunk_size(self):
        self.assertIsInstance(self.check.chunk_size, float)
        self.assertEqual(32.0, self.check.chunk_size)

    def test_has_storage_kws(self):
        self.assertTrue(hasattr(self.check, 'storage_kws'))

    def test_storage_kws(self):
        self.assertDictEqual({}, self.check.storage_kws)

    def test_has_chunk_bytes(self):
        self.assertTrue(hasattr(self.check, 'chunk_bytes'))

    def test_chunk_bytes(self):
        self.assertIsInstance(self.check.chunk_bytes, int)
        self.assertEqual(32 * 1024 * 1024, self.check.chunk_bytes)

    def test_has_fs(self):
        self.assertTrue(hasattr(self.check, 'fs'))

    def test_fs(self):
        self.assertIsInstance(self.check.fs, MemoryFileSystem)

    @patch('swak.pt.train.checkpoint.fsspec.filesystem')
    def test_filesystem_called_with_defaults(self, fs):
        _ = self.check.fs
        fs.assert_called_once_with('memory')

    def test_has_counter(self):
        self.assertTrue(hasattr(self.check, 'counter'))

    def test_counter(self):
        self.assertIsInstance(self.check.counter, int)
        self.assertEqual(0, self.check.counter)

    def test_has_save(self):
        self.assertTrue(hasattr(self.check, 'save'))

    def test_save(self):
        self.assertTrue(callable(self.check.save))

    def test_has_load(self):
        self.assertTrue(hasattr(self.check, 'load'))

    def test_load(self):
        self.assertTrue(callable(self.check.load))

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.check, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.check.reset_parameters))


class TestAttributes(unittest.TestCase):

    def test_path(self):
        check = Checkpoint('/path/to/checkpoint.pt')
        self.assertEqual('/path/to/checkpoint.pt', check.path)

    def test_empty_path_raises(self):
        with self.assertRaises(ValueError):
            _ = Checkpoint('')

    def test_root_path_raises(self):
        with self.assertRaises(ValueError):
            _ = Checkpoint('/checkpoint.pt')

    def test_wrong_path_raises(self):
        with self.assertRaises(TypeError):
            _ = Checkpoint(42)

    def test_path_stripped(self):
        check = Checkpoint('  / path/to/checkpoint.pt  / ')
        self.assertEqual('/path/to/checkpoint.pt', check.path)

    def test_path_prepended(self):
        check = Checkpoint('path/to/checkpoint.pt')
        self.assertEqual('/path/to/checkpoint.pt', check.path)

    def test_storage_str(self):
        check = Checkpoint(storage='file')
        self.assertIsInstance(check.storage, str)
        self.assertEqual(Storage.FILE, check.storage)

    def test_storage_enum(self):
        check = Checkpoint(storage=Storage.FILE)
        self.assertIsInstance(check.storage, str)
        self.assertEqual(Storage.FILE, check.storage)

    def test_storage_raises_wrong_type(self):
        with self.assertRaises(ValueError):
            _ = Checkpoint(storage=42)

    def test_storage_raises_wrong_value(self):
        with self.assertRaises(ValueError):
            _ = Checkpoint(storage='foo')

    def test_chunk_size(self):
        check = Checkpoint(chunk_size=16)
        self.assertEqual(16, check.chunk_size)

    def test_chunk_size_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            _ =  Checkpoint(chunk_size='foo')

    def test_chunk_size_wrong_value_raises(self):
        with self.assertRaises(ValueError):
            _ =  Checkpoint(chunk_size=0)

    def test_storage_kws(self):
        check = Checkpoint(storage_kws={'answer': 42})
        self.assertDictEqual({'answer': 42}, check.storage_kws)

    def test_chunk_bytes_round(self):
        check = Checkpoint(chunk_size=16.2)
        self.assertEqual(64 * 256 * 1024, check.chunk_bytes)
        check = Checkpoint(chunk_size=16.3)
        self.assertEqual(65 * 256 * 1024, check.chunk_bytes)
        check = Checkpoint(chunk_size=16.51)
        self.assertEqual(66 * 256 * 1024, check.chunk_bytes)
        check = Checkpoint(chunk_size=16.8)
        self.assertEqual(67 * 256 * 1024, check.chunk_bytes)
        check = Checkpoint(chunk_size=17.0)
        self.assertEqual(68 * 256 * 1024, check.chunk_bytes)

    def test_fs(self):
        check = Checkpoint(storage=Storage.FILE)
        self.assertIsInstance(check.fs, LocalFileSystem)

    @patch('swak.pt.train.checkpoint.fsspec.filesystem')
    def test_filesystem_called_with_custom(self, fs):
        check = Checkpoint(storage=Storage.FILE, storage_kws={'foo': 'bar'})
        _ = check.fs
        fs.assert_called_once_with('file', foo='bar')


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.epoch = 5
        self.loss = 5.678
        self.model_state_dict = {'model': 'model_state_dict'}
        self.model = Mock()
        self.model.state_dict = Mock(
            return_value=self.model_state_dict
        )
        self.optimizer_state_dict = {'optimizer': 'optimizer_state_dict'}
        self.optimizer = Mock()
        self.optimizer.state_dict = Mock(
            return_value=self.optimizer_state_dict
        )
        self.scheduler_state_dict = {'scheduler': 'scheduler_state_dict'}
        self.scheduler = Mock()
        self.scheduler.state_dict = Mock(
            return_value=self.scheduler_state_dict
        )
        self.state = {
            'epoch': self.epoch,
            'loss': self.loss,
            'model': self.model_state_dict,
            'optimizer': {},
            'scheduler': {},
        }
        self.empty = {
            'epoch': 0,
            'loss': float('inf'),
            'model': OrderedDict(),
            'optimizer': {},
            'scheduler': {}
        }
        self.dir = TemporaryDirectory()
        self.file = self.dir.name + '/checkpoint.pt'
        self.path = Path(self.file)
        self.storage = Storage.FILE

    def tearDown(self):
        self.dir.cleanup()

    @patch('swak.pt.train.checkpoint.uuid.uuid4')
    def test_save_open_called_default(self, uuid):
        uuid.return_value = Mock(hex='hex')
        check = Checkpoint(self.file, self.storage)
        mock_fs = Mock()
        mock_file = mock_open()
        mock_fs.open.return_value = mock_file.return_value

        with patch.object(check, 'fs',  mock_fs):
            check.save(self.epoch, self.loss, self.model)

        mock_fs.open.assert_called_once_with(
            self.file + '.tmp.hex',
            'wb',
            check.chunk_bytes
        )
        mock_fs.move.assert_called_once_with(
            self.file + '.tmp.hex',
            self.file
        )
        mock_fs.rm.assert_not_called()

    @patch('swak.pt.train.checkpoint.uuid.uuid4')
    def test_save_open_called_custom(self, uuid):
        uuid.return_value = Mock(hex='hex')
        check = Checkpoint(self.file, self.storage, chunk_size=16)
        mock_fs = Mock()
        mock_file = mock_open()
        mock_fs.open.return_value = mock_file.return_value

        with patch.object(check, 'fs',  mock_fs):
            check.save(self.epoch, self.loss, self.model)

        mock_fs.open.assert_called_once_with(
            self.file + '.tmp.hex',
            'wb',
            check.chunk_bytes
        )
        mock_fs.move.assert_called_once_with(
            self.file + '.tmp.hex',
            self.file
        )
        mock_fs.rm.assert_not_called()

    @patch('swak.pt.train.checkpoint.uuid.uuid4')
    def test_save_cleans_up_on_yield_error(self, uuid):
        uuid.return_value = Mock(hex='hex')
        check = Checkpoint(self.file, self.storage)
        mock_fs = Mock()
        mock_fs.open.side_effect = PermissionError()

        with self.assertRaises(
                PermissionError
        ), patch.object(
                check, 'fs', mock_fs
        ):
            check.save(self.epoch, self.loss, self.model)

        mock_fs.move.assert_not_called()
        mock_fs.rm.assert_called_once_with(self.file + '.tmp.hex')

    @patch('swak.pt.train.checkpoint.pt.save')
    @patch('swak.pt.train.checkpoint.uuid.uuid4')
    def test_save_cleans_up_on_write_error(self, uuid, save):
        uuid.return_value = Mock(hex='hex')
        check = Checkpoint(self.file, self.storage)
        mock_fs = Mock()
        mock_file = mock_open()
        mock_fs.open.return_value = mock_file.return_value
        save.side_effect = RuntimeError()

        with self.assertRaises(
                RuntimeError
        ), patch.object(
                check, 'fs', mock_fs
        ):
            check.save(self.epoch, self.loss, self.model)

        mock_fs.move.assert_not_called()
        mock_fs.rm.assert_called_once_with(self.file + '.tmp.hex')

    def test_save_model(self):
        check = Checkpoint(self.file, self.storage)
        check.save(self.epoch, self.loss, self.model)
        with self.path.open('rb') as file:
            actual = pt.load(file, weights_only=True)
        self.assertDictEqual(actual, self.state)

    def test_save_model_optimizer(self):
        check = Checkpoint(self.file, self.storage)
        check.save(self.epoch, self.loss, self.model, self.optimizer)
        with self.path.open('rb') as file:
            actual = pt.load(file, weights_only=True)
        expected = self.state | {'optimizer': self.optimizer_state_dict}
        self.assertDictEqual(expected, actual)

    def test_save_model_optimizer_scheduler(self):
        check = Checkpoint(self.file, self.storage)
        check.save(
            self.epoch,
            self.loss,
            self.model,
            self.optimizer,
            self.scheduler
        )
        with self.path.open('rb') as file:
            actual = pt.load(file, weights_only=True)
        expected = self.state | {'optimizer': self.optimizer_state_dict}
        expected = expected | {'scheduler': self.scheduler_state_dict}
        self.assertDictEqual(expected, actual)

    def test_counter_counts(self):
        check = Checkpoint(self.file, self.storage)
        check.save(self.epoch, self.loss, self.model)
        self.assertIsInstance(check.counter, int)
        self.assertEqual(1, check.counter)
        check.save(self.epoch, self.loss, self.model)
        self.assertIsInstance(check.counter, int)
        self.assertEqual(2, check.counter)
        check.save(self.epoch, self.loss, self.model)
        self.assertIsInstance(check.counter, int)
        self.assertEqual(3, check.counter)
        check.save(self.epoch, self.loss, self.model)
        self.assertIsInstance(check.counter, int)
        self.assertEqual(4, check.counter)

    def test_reset_parameters(self):
        check = Checkpoint(self.file, self.storage)
        check.save(self.epoch, self.loss, self.model)
        check.reset_parameters()
        self.assertEqual(0, check.counter)
        with self.path.open('rb') as file:
            actual = pt.load(file, weights_only=True)
        self.assertDictEqual(actual, self.empty)

    @patch('swak.pt.train.checkpoint.pt.load')
    def test_load_open_called_default(self, load):
        check = Checkpoint(self.file, self.storage)
        mock_fs = Mock()
        mock_file = mock_open()
        mock_fs.open.return_value = mock_file.return_value
        load.return_value = self.state

        with patch.object(
                check,
                'fs',
                mock_fs
        ), patch.object(check, 'reset_parameters') as reset_parameters:
            _ = check.load(self.model)

        mock_fs.open.assert_called_once_with(
            self.file,
            'rb',
            check.chunk_bytes
        )
        reset_parameters.assert_not_called()

    @patch('swak.pt.train.checkpoint.pt.load')
    def test_load_open_called_custom(self, load):
        check = Checkpoint(self.file, self.storage, chunk_size=16)
        mock_fs = Mock()
        mock_file = mock_open()
        mock_fs.open.return_value = mock_file.return_value
        load.return_value = self.state

        with patch.object(
                check,
                'fs',
                mock_fs
        ), patch.object(check, 'reset_parameters') as reset_parameters:
            _ = check.load(self.model)

        mock_fs.open.assert_called_once_with(
            self.file,
            'rb',
            check.chunk_bytes
        )
        reset_parameters.assert_not_called()

    def test_load_empty_on_file_not_found(self):
        check = Checkpoint(self.file, self.storage)
        self.assertFalse(self.path.exists())
        epoch, loss = check.load(self.model)
        self.model.load_state_dict.assert_called_once_with(
            self.model_state_dict
        )
        self.assertEqual(0, epoch)
        self.assertEqual(float('inf'), loss)
        with self.path.open('rb') as file:
            actual = pt.load(file, weights_only=True)
        self.assertDictEqual(self.empty, actual)

    @patch('swak.pt.train.checkpoint.pt.load')
    def test_load_error_propagates(self, load):
        with self.path.open('wb') as file:
            pt.save(self.state, file)
        load.side_effect = EOFError()
        check = Checkpoint(self.file, self.storage)
        with self.assertRaises(EOFError):
            _ = check.load(self.model)

    def test_merge_model(self):
        other = {'other': 'other_state_dict'}
        state = self.state | {'model': other}
        with self.path.open('wb') as file:
            pt.save(state, file)
        check = Checkpoint(self.file, self.storage)
        _ = check.load(self.model)
        expected = self.model_state_dict | other
        self.model.load_state_dict.assert_called_once_with(expected)
        self.optimizer.load_state_dict.assert_not_called()
        self.scheduler.load_state_dict.assert_not_called()

    def test_merge_optimizer(self):
        other = {'other': 'other_state_dict'}
        state = self.state | {'optimizer': other}
        with self.path.open('wb') as file:
            pt.save(state, file)
        check = Checkpoint(self.file, self.storage)
        _ = check.load(self.model, self.optimizer)
        expected = self.optimizer_state_dict | other
        self.optimizer.load_state_dict.assert_called_once_with(expected)
        self.scheduler.load_state_dict.assert_not_called()

    def test_merge_scheduler(self):
        other = {'other': 'other_state_dict'}
        state = self.state | {'scheduler': other}
        with self.path.open('wb') as file:
            pt.save(state, file)
        check = Checkpoint(self.file, self.storage)
        _ = check.load(self.model, self.optimizer, self.scheduler)
        expected = self.scheduler_state_dict | other
        self.scheduler.load_state_dict.assert_called_once_with(expected)

    def test_return_values(self):
        with self.path.open('wb') as file:
            pt.save(self.state, file)
        check = Checkpoint(self.file, self.storage)
        epoch, loss = check.load(self.model, self.optimizer, self.scheduler)
        self.assertEqual(self.epoch, epoch)
        self.assertEqual(self.loss, loss)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        check = Checkpoint()
        expected = "Checkpoint('/tmp/checkpoint.pt', 'memory', 32.0, {})"
        self.assertEqual(expected, repr(check))

    def test_custom_repr(self):
        check = Checkpoint(
            '/path/to/file.pt',
            'file',
            16,
            {'foo': 'bar'}
        )
        exp = "Checkpoint('/path/to/file.pt', 'file', 16.0, {'foo': 'bar'})"
        self.assertEqual(exp, repr(check))

    def test_pickle_works(self):
        check = Checkpoint(
            '/path/to/file.pt',
            'file',
            16,
            {'foo': 'bar'}
        )
        _ = pickle.loads(pickle.dumps(check))


if __name__ == '__main__':
    unittest.main()
