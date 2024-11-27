import unittest
import torch as pt
from swak.pt.losses import XEntropyLoss
from torch.nn import CrossEntropyLoss


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.original = CrossEntropyLoss()
        self.subclass = XEntropyLoss()

    def test_has_weight(self):
        self.assertTrue(hasattr(self.subclass, 'weight'))

    def test_weight(self):
        self.assertIs(self.original.weight, self.subclass.weight)

    def test_has_ignore_index(self):
        self.assertTrue(hasattr(self.subclass, 'ignore_index'))

    def test_ignore_index(self):
        self.assertEqual(
            self.original.ignore_index,
            self.subclass.ignore_index
        )

    def test_has_reduction(self):
        self.assertTrue(hasattr(self.subclass, 'reduction'))

    def test_reduction(self):
        self.assertEqual(self.original.reduction, self.subclass.reduction)

    def test_has_label_smoothing(self):
        self.assertTrue(hasattr(self.subclass, 'label_smoothing'))

    def test_label_smoothing(self):
        self.assertEqual(
            self.original.label_smoothing,
            self.subclass.label_smoothing
        )


class TestAttributes(unittest.TestCase):

    def setUp(self):
        weight =  pt.rand(5)
        self.original = CrossEntropyLoss(
            weight=weight,
            ignore_index=32,
            reduction='sum',
            label_smoothing=0.3
        )
        self.subclass = XEntropyLoss(weight, 32, 'sum', 0.3)

    def test_weight(self):
        self.assertIs(self.original.weight, self.subclass.weight)

    def test_ignore_index(self):
        self.assertEqual(
            self.original.ignore_index,
            self.subclass.ignore_index
        )

    def test_reduction(self):
        self.assertEqual(self.original.reduction, self.subclass.reduction)

    def test_label_smoothing(self):
        self.assertEqual(
            self.original.label_smoothing,
            self.subclass.label_smoothing
        )


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.inp = pt.rand(32, 128)
        self.tgt = pt.randint(0, 128, (32, ))
        self.ls = 0.2
        self.original00 = CrossEntropyLoss()
        self.subclass00 = XEntropyLoss()
        self.original02 = CrossEntropyLoss(label_smoothing=self.ls)
        self.subclass02 = XEntropyLoss(label_smoothing=self.ls)

    def test_no_toggle_00(self):
        expected = self.original00(self.inp, self.tgt)
        actual = self.subclass00(self.inp, self.tgt)
        pt.testing.assert_close(actual, expected)

    def test_no_toggle_02(self):
        expected = self.original02(self.inp, self.tgt)
        actual = self.subclass02(self.inp, self.tgt)
        pt.testing.assert_close(actual, expected)

    def test_toggle_train_00(self):
        self.original00.train()
        self.subclass00.train()
        self.assertEqual(
            self.original00.label_smoothing,
            self.subclass00.label_smoothing
        )
        expected = self.original00(self.inp, self.tgt)
        actual = self.subclass00(self.inp, self.tgt)
        pt.testing.assert_close(actual, expected)

    def test_toggle_train_true_00(self):
        self.original00.train(True)
        self.subclass00.train(True)
        self.assertEqual(
            self.original00.label_smoothing,
            self.subclass00.label_smoothing
        )
        expected = self.original00(self.inp, self.tgt)
        actual = self.subclass00(self.inp, self.tgt)
        pt.testing.assert_close(actual, expected)

    def test_toggle_train_02(self):
        self.original02.train()
        self.subclass02.train()
        self.assertEqual(
            self.original00.label_smoothing,
            self.subclass00.label_smoothing
        )
        expected = self.original02(self.inp, self.tgt)
        actual = self.subclass02(self.inp, self.tgt)
        pt.testing.assert_close(actual, expected)

    def test_toggle_train_true_02(self):
        self.original02.train(True)
        self.subclass02.train(True)
        self.assertEqual(
            self.original00.label_smoothing,
            self.subclass00.label_smoothing
        )
        expected = self.original02(self.inp, self.tgt)
        actual = self.subclass02(self.inp, self.tgt)
        pt.testing.assert_close(actual, expected)

    def test_toggle_eval_00(self):
        self.original00.eval()
        self.subclass00.eval()
        self.assertEqual(
            self.original00.label_smoothing,
            self.subclass00.label_smoothing
        )
        expected = self.original00(self.inp, self.tgt)
        actual = self.subclass00(self.inp, self.tgt)
        pt.testing.assert_close(actual, expected)

    def test_toggle_eval_02(self):
        self.original02.eval()
        self.subclass02.eval()
        self.assertNotEqual(
            self.original02.label_smoothing,
            self.subclass02.label_smoothing
        )
        self.assertEqual(self.ls, self.original02.label_smoothing)
        self.assertEqual(0.0, self.subclass02.label_smoothing)
        expected = self.original02(self.inp, self.tgt)
        actual = self.subclass02(self.inp, self.tgt)
        self.assertNotEqual(expected, actual)

    def test_toggle_train_false_00(self):
        self.original00.train(False)
        self.subclass00.train(False)
        self.assertEqual(
            self.original00.label_smoothing,
            self.subclass00.label_smoothing
        )
        expected = self.original00(self.inp, self.tgt)
        actual = self.subclass00(self.inp, self.tgt)
        pt.testing.assert_close(actual, expected)

    def test_toggle_train_false_02(self):
        self.original02.train(False)
        self.subclass02.train(False)
        self.assertNotEqual(
            self.original02.label_smoothing,
            self.subclass02.label_smoothing
        )
        self.assertEqual(self.ls, self.original02.label_smoothing)
        self.assertEqual(0.0, self.subclass02.label_smoothing)
        expected = self.original02(self.inp, self.tgt)
        actual = self.subclass02(self.inp, self.tgt)
        self.assertNotEqual(expected, actual)

    def test_toggle_back_and_forth(self):
        self.original02.train()
        self.subclass02.train()
        self.assertEqual(
            self.original00.label_smoothing,
            self.subclass00.label_smoothing
        )
        expected = self.original02(self.inp, self.tgt)
        actual = self.subclass02(self.inp, self.tgt)
        pt.testing.assert_close(actual, expected)

        self.original02.train(False)
        self.subclass02.train(False)
        self.assertNotEqual(
            self.original02.label_smoothing,
            self.subclass02.label_smoothing
        )
        self.assertEqual(self.ls, self.original02.label_smoothing)
        self.assertEqual(0.0, self.subclass02.label_smoothing)
        expected = self.original02(self.inp, self.tgt)
        actual = self.subclass02(self.inp, self.tgt)
        self.assertNotEqual(expected, actual)

        self.original02.train(True)
        self.subclass02.train(True)
        self.assertEqual(
            self.original00.label_smoothing,
            self.subclass00.label_smoothing
        )
        expected = self.original02(self.inp, self.tgt)
        actual = self.subclass02(self.inp, self.tgt)
        pt.testing.assert_close(actual, expected)

        self.original02.eval()
        self.subclass02.eval()
        self.assertNotEqual(
            self.original02.label_smoothing,
            self.subclass02.label_smoothing
        )
        self.assertEqual(self.ls, self.original02.label_smoothing)
        self.assertEqual(0.0, self.subclass02.label_smoothing)
        expected = self.original02(self.inp, self.tgt)
        actual = self.subclass02(self.inp, self.tgt)
        self.assertNotEqual(expected, actual)

        self.original02.train()
        self.subclass02.train()
        self.assertEqual(
            self.original00.label_smoothing,
            self.subclass00.label_smoothing
        )
        expected = self.original02(self.inp, self.tgt)
        actual = self.subclass02(self.inp, self.tgt)
        pt.testing.assert_close(actual, expected)


class TestMisc(unittest.TestCase):

    def test_subclassing(self):
        loss = XEntropyLoss()
        self.assertTrue(issubclass(XEntropyLoss, CrossEntropyLoss))
        self.assertTrue(issubclass(XEntropyLoss, pt.nn.Module))
        self.assertIsInstance(loss, CrossEntropyLoss)
        self.assertIsInstance(loss, pt.nn.Module)


if __name__ == '__main__':
    unittest.main()
