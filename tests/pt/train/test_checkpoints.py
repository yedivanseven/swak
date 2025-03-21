import unittest
from unittest.mock import patch, Mock
from collections import OrderedDict
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import torch as pt
from swak.pt.train import Checkpoint, InMemory, OnDisk, State


class Check(Checkpoint):

    def _save_state(self, state: State) -> None:
        pass

    def _load_state(self) -> State:
        return {
            'epoch': 42,
            'loss': 1.234,
            'model': OrderedDict(foo=1, bar=2),
            'optimizer': {'a': 3},
            'scheduler': {'b': 4}
        }


class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        self.check = Check()
        self.epoch = 5
        self.loss = 5.678
        self.model = Mock()
        self.model.state_dict = Mock(
            return_value={'model': 'model_state_dict'}
        )
        self.model.load_state_dict = Mock()
        self.optimizer = Mock()
        self.optimizer.state_dict = Mock(
            return_value={'optimizer': 'optimizer_state_dict'}
        )
        self.optimizer.load_state_dict = Mock()
        self.scheduler = Mock()
        self.scheduler.state_dict = Mock(
            return_value={'scheduler': 'scheduler_state_dict'}
        )
        self.scheduler.load_state_dict = Mock()

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

    def test_counter_counts(self):
        self.check.save(self.epoch, self.loss, self.model)
        self.assertEqual(1, self.check.counter)
        self.check.save(self.epoch, self.loss, self.model)
        self.assertEqual(2, self.check.counter)
        self.check.save(self.epoch, self.loss, self.model)
        self.assertEqual(3, self.check.counter)
        self.check.save(self.epoch, self.loss, self.model)
        self.assertEqual(4, self.check.counter)

    def test_reset_parameters_resets(self):
        for _ in range(7):
            self.check.save(self.epoch, self.loss, self.model)
        with patch.object(self.check, '_save_state') as mock:
            self.check.reset_parameters()
            mock.assert_called_once_with({
                'epoch': 0,
                'loss': float('inf'),
                'model': OrderedDict(),
                'optimizer': {},
                'scheduler': {}
            })
        self.assertEqual(0, self.check.counter)

    def test_save_calls_save_state_model(self):
        with patch.object(self.check, '_save_state') as mock:
            self.check.save(1, 3.45, self.model)
            self.model.state_dict.assert_called_once_with()
            mock.assert_called_once_with({
                'epoch': 1,
                'loss': 3.45,
                'model': {'model': 'model_state_dict'},
                'optimizer': {},
                'scheduler': {}
            })

    def test_save_calls_save_state_optimizer(self):
        with patch.object(self.check, '_save_state') as mock:
            self.check.save(1, 3.45, self.model, self.optimizer)
            self.optimizer.state_dict.assert_called_once_with()
            mock.assert_called_once_with({
                'epoch': 1,
                'loss': 3.45,
                'model': {'model': 'model_state_dict'},
                'optimizer': {'optimizer': 'optimizer_state_dict'},
                'scheduler': {}
            })

    def test_save_calls_save_state_scheduler(self):
        with patch.object(self.check, '_save_state') as mock:
            self.check.save(
                1,
                3.45,
                self.model,
                self.optimizer,
                self.scheduler
            )
            self.scheduler.state_dict.assert_called_once_with()
            mock.assert_called_once_with({
                'epoch': 1,
                'loss': 3.45,
                'model': {'model': 'model_state_dict'},
                'optimizer': {'optimizer': 'optimizer_state_dict'},
                'scheduler': {'scheduler': 'scheduler_state_dict'}
            })

    def test_load_calls_load_state_model(self):
        epoch, loss = self.check.load(self.model)
        self.assertEqual(42, epoch)
        self.assertEqual(1.234, loss)
        self.model.load_state_dict.assert_called_once_with({
            'model': 'model_state_dict',
            'foo': 1,
            'bar': 2
        })

    def test_load_calls_load_state_optimizer(self):
        epoch, loss = self.check.load(self.model, self.optimizer)
        self.assertEqual(42, epoch)
        self.assertEqual(1.234, loss)
        self.optimizer.load_state_dict.assert_called_once_with({
            'optimizer': 'optimizer_state_dict',
            'a': 3
        })

    def test_load_calls_load_state_scheduler(self):
        epoch, loss = self.check.load(
            self.model,
            self.optimizer,
            self.scheduler
        )
        self.assertEqual(42, epoch)
        self.assertEqual(1.234, loss)
        self.scheduler.load_state_dict.assert_called_once_with({
            'scheduler': 'scheduler_state_dict',
            'b': 4
        })


class TestInMemory(unittest.TestCase):

    def setUp(self):
        self.check = InMemory()

    def test_has_state(self):
        self.assertTrue(hasattr(self.check, 'state'))

    def test_state(self):
        expected = {
            'epoch': 0,
            'loss': float('inf'),
            'model': OrderedDict(),
            'optimizer': {},
            'scheduler': {}
        }
        self.assertDictEqual(expected, self.check.state)

    def test_repr(self):
        expected = 'InMemory()'
        self.assertEqual(expected, repr(self.check))

    def test_save_copies_and_sets_target_on_state(self):
        self.assertFalse(hasattr(pt.ones(1), 'target'))
        model = Mock()
        a = pt.ones(1, device='cpu')
        c = pt.ones(2, device='cpu')
        d = pt.ones(3, device='cpu')
        e = pt.ones(4, device='cpu')
        model.state_dict = Mock(return_value=OrderedDict(
            a=a,
            b=OrderedDict(c=c, d=d),
            e=[e]
        ))
        cpu = pt.device('cpu')
        self.check.save(5, 6.7, model)
        self.assertEqual(cpu, self.check.state['model']['a'].target)
        self.assertIsNot(a, self.check.state['model']['a'])
        self.assertEqual(cpu, self.check.state['model']['b']['c'].target)
        self.assertIsNot(c, self.check.state['model']['b']['c'])
        self.assertEqual(cpu, self.check.state['model']['b']['d'].target)
        self.assertIsNot(d, self.check.state['model']['b']['d'])
        self.assertEqual(cpu, self.check.state['model']['e'][0].target)
        self.assertIsNot(e, self.check.state['model']['e'][0])

    def test_load_copies_and_restores_target(self):
        self.assertFalse(hasattr(pt.ones(1), 'target'))
        model = Mock()
        a = pt.ones(1, device='cpu')
        c = pt.ones(1, device='cpu')
        d = pt.ones(1, device='cpu')
        e = pt.ones(1, device='cpu')
        model.state_dict = Mock(return_value=OrderedDict(
            a=a,
            b=OrderedDict(c=c, d=d),
            e=[e]
        ))
        self.check.save(5, 6.7, model)
        self.check.state['model']['a'].target = 'cuda'
        self.check.state['model']['b']['c'].target = 'cuda'
        self.check.state['model']['b']['d'].target = 'cuda'
        self.check.state['model']['e'][0].target = 'cuda'
        with patch('torch.empty_like', return_value=pt.ones(1)) as mock:
            self.check.load(model)
            self.assertEqual(4, mock.call_count)
            self.assertDictEqual({'device': 'cuda'}, mock.call_args[1])

    def test_load_called_before_save(self):
        model = pt.nn.Linear(1, 1)
        optimizer = pt.optim.Adam(model.parameters())
        scheduler = pt.optim.lr_scheduler.ConstantLR(optimizer)
        epoch, loss = self.check.load(model, optimizer, scheduler)
        self.assertIsInstance(epoch, int)
        self.assertEqual(0, epoch)
        self.assertEqual(float('inf'), loss)


class TestOnDisk(unittest.TestCase):

    def setUp(self):
        with NamedTemporaryFile() as file:
            self.path = file.name
        self.check = OnDisk(self.path)
        self.model = Mock()
        self.model.state_dict = Mock(
            return_value={'model': 'model_state_dict'}
        )

    def test_has_path(self):
        self.assertTrue(hasattr(self.check, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.check.path)

    def test_path_stripped(self):
        path = '  /path '
        check = OnDisk(path)
        self.assertEqual('/path', check.path)

    def test_path_like(self):
        check = OnDisk(Path(self.path))
        self.assertEqual(self.path, check.path)

    def test_has_default_create(self):
        self.assertTrue(hasattr(self.check, 'create'))

    def test_default_create(self):
        self.assertIsInstance(self.check.create, bool)
        self.assertFalse(self.check.create)

    def test_custom_create(self):
        check = OnDisk(self.path, True)
        self.assertIsInstance(check.create, bool)
        self.assertTrue(check.create)

    @patch('torch.save')
    def test_save_called_on_save(self, mock):
        self.check.save(1, 2.3, self.model)
        mock.assert_called_once_with({
            'epoch': 1,
            'loss': 2.3,
            'model': {'model': 'model_state_dict'},
            'optimizer': {},
            'scheduler': {}
        },
            self.path
        )

    @patch('torch.load')
    def test_load_called_on_load(self, mock):
        _ = self.check.load(self.model)
        mock.assert_called_once_with(self.path, weights_only=True)

    def test_save_raises_default(self):
        model = pt.nn.Linear(1, 1)
        with TemporaryDirectory() as base:
            file = base + '/path/to/checkpoint.pt'
            check = OnDisk(file)
            with self.assertRaises(RuntimeError):
                check.save(1, 1.0, model)

    def test_save_works_custom(self):
        model = pt.nn.Linear(1, 1)
        with TemporaryDirectory() as base:
            file = base + '/path/to/checkpoint.pt'
            check = OnDisk(file, create=True)
            check.save(1, 1.0, model)

    def test_load_called_before_save_raises(self):
        model = pt.nn.Linear(1, 1)
        optimizer = pt.optim.Adam(model.parameters())
        scheduler = pt.optim.lr_scheduler.ConstantLR(optimizer)
        with NamedTemporaryFile() as file:
            check = OnDisk(file, not_found='raise')
            with self.assertRaises(FileNotFoundError):
                _ = check.load(model, optimizer, scheduler)
        with NamedTemporaryFile() as file:
            check = OnDisk(file.name, not_found='raise')
            with self.assertRaises(EOFError):
                _ = check.load(model, optimizer, scheduler)

    def test_load_called_before_save_warns(self):
        model = pt.nn.Linear(1, 1)
        optimizer = pt.optim.Adam(model.parameters())
        scheduler = pt.optim.lr_scheduler.ConstantLR(optimizer)
        with NamedTemporaryFile() as file:
            check = OnDisk(file.name, not_found='warn')
            with self.assertWarns(UserWarning):
                epoch, loss = check.load(model, optimizer, scheduler)
        self.assertIsInstance(epoch, int)
        self.assertEqual(0, epoch)
        self.assertEqual(float('inf'), loss)

    def test_load_called_before_save_works_silently(self):
        model = pt.nn.Linear(1, 1)
        optimizer = pt.optim.Adam(model.parameters())
        scheduler = pt.optim.lr_scheduler.ConstantLR(optimizer)
        with NamedTemporaryFile() as file:
            check = OnDisk(file.name, not_found='ignore')
            epoch, loss = check.load(model, optimizer, scheduler)
        self.assertIsInstance(epoch, int)
        self.assertEqual(0, epoch)
        self.assertEqual(float('inf'), loss)

    def test_default_repr(self):
        expected = f"OnDisk('{self.path}', False)"
        self.assertEqual(expected, repr(self.check))

    def test_custom_repr(self):
        check = OnDisk(self.path, True)
        expected = f"OnDisk('{self.path}', True)"
        self.assertEqual(expected, repr(check))


if __name__ == '__main__':
    unittest.main()
