import unittest
from unittest.mock import Mock
from torch.nn.modules.module import _IncompatibleKeys
from swak.pt.train.schedulers import NoSchedule


class TestNoSchedule(unittest.TestCase):

    def setUp(self):
        self.optimizer = Mock()
        self.expected = 42
        self.optimizer.defaults.__getitem__ = Mock(return_value=self.expected)
        self.schedule = NoSchedule(self.optimizer)

    def test_takes_args_kwargs(self):
        _ = NoSchedule(self.optimizer, 'foo', 1, bar=2, baz=3)

    def test_has_optimizer(self):
        self.assertTrue(hasattr(self.schedule, 'optimizer'))

    def test_optimizer(self):
        self.assertIs(self.schedule.optimizer, self.optimizer)

    def test_has_step(self):
        self.assertTrue(hasattr(self.schedule, 'step'))

    def test_step(self):
        self.assertTrue(callable(self.schedule.step))

    def test_step_takes_args_kwargs(self):
        self.schedule.step('foo', 1, bar=2, baz=3)

    def test_has_get_last_lr(self):
        self.assertTrue(hasattr(self.schedule, 'get_last_lr'))

    def test_get_last_lr(self):
        self.assertTrue(callable(self.schedule.get_last_lr))

    def test_call_get_last_lr(self):
        actual = self.schedule.get_last_lr()
        self.assertListEqual([self.expected], actual)

    def test_step_does_nothing(self):
        for _ in range(1000):
            self.schedule.step()
        actual = self.schedule.get_last_lr()
        self.assertListEqual([self.expected], actual)

    def test_has_state_dict(self):
        self.assertTrue(hasattr(self.schedule, 'state_dict'))

    def test_state_dict(self):
        self.assertTrue(callable(self.schedule.state_dict))

    def test_call_state_dict(self):
        actual = self.schedule.state_dict()
        self.assertDictEqual({}, actual)

    def test_has_load_state_dict(self):
        self.assertTrue(hasattr(self.schedule, 'load_state_dict'))

    def test_load_state_dict(self):
        self.assertTrue(callable(self.schedule.load_state_dict))

    def test_call_load_state_dict(self):
        actual = self.schedule.load_state_dict('foo', 1, bar=2)
        self.assertIsInstance(actual, _IncompatibleKeys)
        self.assertListEqual([], actual.missing_keys)
        self.assertListEqual([], actual.unexpected_keys)


if __name__ == '__main__':
    unittest.main()
