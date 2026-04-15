import math
import pickle
import unittest
import torch.nn as ptn
import torch.optim as pto
import torch.optim.lr_scheduler as ptos
from swak.funcflow import Curry
from swak.pt.train.schedulers import NoSchedule
from swak.pt.train import LinearInverse, LinearExponential, LinearCosine


class TestNoSchedule(unittest.TestCase):

    def test_is_curry(self):
        self.assertIsInstance(NoSchedule, Curry)

    def test_instance_type(self):
        model = ptn.Linear(1, 1)
        optimizer = pto.Adam(model.parameters())
        scheduler = NoSchedule(optimizer)
        self.assertIsInstance(scheduler, ptos.LambdaLR)

    def test_initial_lr(self):
        model = ptn.Linear(1, 1)
        optimizer = pto.Adam(model.parameters())
        scheduler = NoSchedule(optimizer)
        actual = scheduler.get_last_lr()
        self.assertListEqual([optimizer.defaults['lr']], actual)

    def test_does_nothing(self):
        model = ptn.Linear(1, 1)
        optimizer = pto.Adam(model.parameters())
        scheduler = NoSchedule(optimizer)
        for _ in range(1000):
            optimizer.step()
            scheduler.step()
        actual = scheduler.get_last_lr()
        self.assertListEqual([optimizer.defaults['lr']], actual)


class TestLinearInverse(unittest.TestCase):

    def setUp(self):
        self.default = LinearInverse()
        self.ramp = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.warmup = len(self.ramp)
        self.power = 0.7
        self.custom = LinearInverse(self.warmup, self.power)

    def test_default_has_warmup(self):
        self.assertTrue(hasattr(self.default, 'warmup'))

    def test_default_warmup(self):
        self.assertIsInstance(self.default.warmup, int)
        self.assertEqual(0, self.default.warmup)

    def test_default_has_power(self):
        self.assertTrue(hasattr(self.default, 'power'))

    def test_default_power(self):
        self.assertEqual(0.5, self.default.power)

    def test_custom_warmup(self):
        self.assertIsInstance(self.custom.warmup, int)
        self.assertEqual(self.warmup, self.custom.warmup)

    def test_custom_power(self):
        self.assertEqual(self.power, self.custom.power)

    def test_warmup_at_least_zero(self):
        scheduler = LinearInverse(-1)
        self.assertEqual(0, scheduler.warmup)

    def test_power_positive(self):
        scheduler = LinearInverse(power=-1.23)
        self.assertEqual(0.5, scheduler.power)

    def test_power_at_least_05(self):
        scheduler = LinearInverse(power=0.123)
        self.assertEqual(0.5, scheduler.power)

    def test_power_at_most_1(self):
        scheduler = LinearInverse(power=1.123)
        self.assertEqual(1.0, scheduler.power)

    def test_callable(self):
        self.assertTrue(callable(self.default))

    def test_ramp_up(self):
        for epoch in range(self.warmup):
            self.assertEqual(self.ramp[epoch], self.custom(epoch))
        self.assertEqual(1.0, self.custom(self.warmup))

    def test_scale_down(self):
        for epoch in range(self.warmup + 1, self.warmup + 11):
            expected = (epoch - self.warmup) ** -self.power
            self.assertEqual(expected, self.custom(epoch))

    def test_no_warmup(self):
        scheduler = LinearInverse(0, self.power)
        self.assertEqual(1.0, scheduler(0))
        for epoch in range(1, 11):
            expected = epoch ** -self.power
            self.assertEqual(expected, scheduler(epoch))

    def test_repr(self):
        expected = 'LinearInverse(5, 0.7)'
        self.assertEqual(expected, repr(self.custom))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.custom))


class TestLinearExponential(unittest.TestCase):

    def setUp(self):
        self.default = LinearExponential()
        self.ramp = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.warmup = len(self.ramp)
        self.gamma = 0.7
        self.custom = LinearExponential(self.warmup, self.gamma)

    def test_default_has_warmup(self):
        self.assertTrue(hasattr(self.default, 'warmup'))

    def test_default_warmup(self):
        self.assertIsInstance(self.default.warmup, int)
        self.assertEqual(0, self.default.warmup)

    def test_default_has_gamma(self):
        self.assertTrue(hasattr(self.default, 'gamma'))

    def test_default_gamma(self):
        self.assertEqual(0.95, self.default.gamma)

    def test_custom_warmup(self):
        self.assertIsInstance(self.custom.warmup, int)
        self.assertEqual(self.warmup, self.custom.warmup)

    def test_custom_gamma(self):
        self.assertEqual(self.gamma, self.custom.gamma)

    def test_warmup_at_least_zero(self):
        scheduler = LinearExponential(-3)
        self.assertEqual(0, scheduler.warmup)

    def test_gamma_at_least_0(self):
        scheduler = LinearExponential(gamma=-0.123)
        self.assertEqual(0.0, scheduler.gamma)

    def test_gamma_at_most_1(self):
        scheduler = LinearExponential(gamma=1.123)
        self.assertEqual(1.0, scheduler.gamma)

    def test_callable(self):
        self.assertTrue(callable(self.default))

    def test_ramp_up(self):
        for epoch in range(self.warmup):
            self.assertEqual(self.ramp[epoch], self.custom(epoch))
        self.assertEqual(1.0, self.custom(self.warmup))

    def test_scale_down(self):
        for epoch in range(self.warmup + 1, self.warmup + 10):
            expected = self.gamma ** (epoch - self.warmup)
            self.assertEqual(expected, self.custom(epoch))

    def test_no_warmup(self):
        scheduler = LinearExponential(0, self.gamma)
        self.assertEqual(1.0, scheduler(0))
        for epoch in range(1, 11):
            expected = self.gamma ** epoch
            self.assertEqual(expected, scheduler(epoch))

    def test_repr(self):
        expected = 'LinearExponential(5, 0.7)'
        self.assertEqual(expected, repr(self.custom))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.custom))


class TestLinearCosine(unittest.TestCase):

    def setUp(self):
        self.default = LinearCosine()
        self.ramp = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.warmup = len(self.ramp)
        self.cooldown = 13
        self.custom = LinearCosine(self.warmup, self.cooldown)

    def test_default_has_warmup(self):
        self.assertTrue(hasattr(self.default, 'warmup'))

    def test_default_warmup(self):
        self.assertIsInstance(self.default.warmup, int)
        self.assertEqual(0, self.default.warmup)

    def test_default_has_cooldown(self):
        self.assertTrue(hasattr(self.default, 'cooldown'))

    def test_default_cooldown(self):
        self.assertEqual(100, self.default.cooldown)

    def test_custom_warmup(self):
        self.assertIsInstance(self.custom.warmup, int)
        self.assertEqual(self.warmup, self.custom.warmup)

    def test_custom_gamma(self):
        self.assertEqual(self.cooldown, self.custom.cooldown)

    def test_warmup_at_least_zero(self):
        scheduler = LinearCosine(-3)
        self.assertEqual(0, scheduler.warmup)

    def test_cooldown_at_least_1(self):
        scheduler = LinearCosine(cooldown=-123)
        self.assertEqual(1, scheduler.cooldown)

    def test_callable(self):
        self.assertTrue(callable(self.default))

    def test_ramp_up(self):
        for epoch in range(self.warmup):
            self.assertEqual(self.ramp[epoch], self.custom(epoch))
        self.assertEqual(1.0, self.custom(self.warmup))

    def test_scale_down(self):
        for epoch in range(self.warmup + 1, self.warmup + self.cooldown):
            arg = math.pi * (epoch - self.warmup) / self.cooldown
            expected = 0.5 + 0.5 * math.cos(arg)
            self.assertEqual(expected, self.custom(epoch))

    def test_beyond_scale_down(self):
        for epoch in range(
                self.warmup + self.cooldown,
                self.warmup + self.cooldown + 20
        ):
            arg = math.pi * (self.cooldown - 1) / self.cooldown
            expected = 0.5 + 0.5 * math.cos(arg)
            self.assertEqual(expected, self.custom(epoch))

    def test_no_warmup(self):
        scheduler = LinearCosine(0, self.cooldown)
        self.assertEqual(1.0, scheduler(0))
        for epoch in range(1, self.cooldown):
            arg = math.pi * epoch / self.cooldown
            expected = 0.5 + 0.5 * math.cos(arg)
            self.assertEqual(expected, scheduler(epoch))

    def test_cooldown_1(self):
        scheduler = LinearCosine(self.warmup, 1)
        for epoch in range(self.warmup, self.warmup + 1):
            self.assertEqual(1.0, scheduler(epoch))

    def test_no_warmup_cooldown_1(self):
        scheduler = LinearCosine(0, 1)
        for epoch in range(23):
            self.assertEqual(1.0, scheduler(epoch))

    def test_repr(self):
        expected = 'LinearCosine(5, 13)'
        self.assertEqual(expected, repr(self.custom))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.custom))


if __name__ == '__main__':
    unittest.main()
