import unittest
from unittest.mock import Mock
from swak.pt.train import StepPrinter, EpochPrinter, TrainPrinter


class TestStepPrinter(unittest.TestCase):

    def setUp(self):
        self.print = StepPrinter()

    def test_has_printer(self):
        self.assertTrue(hasattr(self.print, 'printer'))

    def test_default_printer(self):
        self.assertIs(self.print.printer, print)

    def test_custom_printer(self):
        printer = object()
        step_cb = StepPrinter(printer)
        self.assertIs(step_cb.printer, printer)

    def test_has_sep(self):
        self.assertTrue(hasattr(self.print, 'sep'))

    def test_default_sep(self):
        self.assertEqual(',', self.print.sep)

    def test_custom_sep(self):
        step_cb = StepPrinter(sep=',')
        self.assertEqual(',', step_cb.sep)

    def test_default_repr(self):
        expected = "StepPrinter(print, ',')"
        self.assertEqual(expected, repr(self.print))

    def test_custom_repr(self):

        def f(_):
            pass

        step_cb = StepPrinter(f, ', ')
        exp = "StepPrinter(TestStepPrinter.test_custom_repr.<locals>.f, ', ')"
        self.assertEqual(exp, repr(step_cb))

    def test_callable(self):
        self.assertTrue(callable(self.print))

    def test_print_called_with_default_sep(self):
        mock = Mock()
        step_cb = StepPrinter(mock)
        step_cb(0.2, 0.01, 0.3)
        mock.assert_called_once()
        expected = '0.20000,0.01000,0.30000'
        actual = mock.call_args[0][0]
        self.assertEqual(expected, actual)

    def test_print_called_with_custom_sep(self):
        mock = Mock()
        step_cb = StepPrinter(mock, ';')
        step_cb(0.2, 0.01, 0.3)
        mock.assert_called_once()
        expected = '0.20000;0.01000;0.30000'
        actual = mock.call_args[0][0]
        self.assertEqual(expected, actual)

    def test_iterable(self):
        printers = list(self.print)
        self.assertEqual(1, len(printers))
        self.assertIs(printers[0], self.print)


class TestEpochPrinter(unittest.TestCase):

    def setUp(self):
        self.print = EpochPrinter()

    def test_has_printer(self):
        self.assertTrue(hasattr(self.print, 'printer'))

    def test_default_printer(self):
        self.assertIs(self.print.printer, print)

    def test_custom_printer(self):
        printer = object()
        epoch_cb = EpochPrinter(printer)
        self.assertIs(epoch_cb.printer, printer)

    def test_has_close(self):
        self.assertTrue(hasattr(self.print, 'close'))

    def test_close(self):
        self.assertTrue(callable(self.print.close))

    def test_call_close(self):
        self.print.close()

    def test_default_repr(self):
        expected = 'EpochPrinter(print)'
        self.assertEqual(expected, repr(self.print))

    def test_custom_repr(self):

        def f(_):
            pass

        epoch_cb = EpochPrinter(f)
        expected = 'EpochPrinter(TestEpochPrinter.test_custom_repr.<locals>.f)'
        self.assertEqual(expected, repr(epoch_cb))

    def test_callable(self):
        self.assertTrue(callable(self.print))

    def test_print_called_with_str(self):
        mock = Mock()
        epoch_cb = EpochPrinter(mock)
        epoch_cb(
            1,
            0.1,
            0.2,
            0.3,
            'model',
            'data'
        )
        mock.assert_called_once()
        actual = mock.call_args[0][0]
        self.assertIsInstance(actual, str)

    def test_iterable(self):
        printers = list(self.print)
        self.assertEqual(1, len(printers))
        self.assertIs(printers[0], self.print)


class TestTrainPrinter(unittest.TestCase):

    def setUp(self):
        self.print = TrainPrinter()

    def test_has_printer(self):
        self.assertTrue(hasattr(self.print, 'printer'))

    def test_default_printer(self):
        self.assertIs(self.print.printer, print)

    def test_custom_printer(self):
        printer = object()
        epoch_cb = TrainPrinter(printer)
        self.assertIs(epoch_cb.printer, printer)

    def test_default_repr(self):
        expected = 'TrainPrinter(print)'
        self.assertEqual(repr(self.print), expected)

    def test_custom_repr(self):

        def f(_):
            pass

        epoch_cb = TrainPrinter(f)
        expected = 'TrainPrinter(TestTrainPrinter.test_custom_repr.<locals>.f)'
        self.assertEqual(repr(epoch_cb), expected)

    def test_callable(self):
        self.assertTrue(callable(self.print))

    def test_print_called_with_str_false(self):
        mock = Mock()
        epoch_cb = TrainPrinter(mock)
        epoch_cb(
            1,
            2,
            0.3,
            False,
            'history'
        )
        mock.assert_called_once()
        actual = mock.call_args[0][0]
        self.assertIsInstance(actual, str)

    def test_print_called_with_str_true(self):
        mock = Mock()
        epoch_cb = TrainPrinter(mock)
        epoch_cb(
            1,
            2,
            0.3,
            True,
            'history'
        )
        mock.assert_called_once()
        actual = mock.call_args[0][0]
        self.assertIsInstance(actual, str)

    def test_iterable(self):
        printers = list(self.print)
        self.assertEqual(1, len(printers))
        self.assertIs(printers[0], self.print)


if __name__ == '__main__':
    unittest.main()
