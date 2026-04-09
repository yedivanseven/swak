import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import AsFreq


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.freq = '1D'
        self.asfreq = AsFreq(self.freq)

    def test_has_freq(self):
        self.assertTrue(hasattr(self.asfreq, 'freq'))

    def test_freq(self):
        self.assertEqual(self.freq, self.asfreq.freq)

    def test_has_method(self):
        self.assertTrue(hasattr(self.asfreq, 'method'))

    def test_method(self):
        self.assertIsNone(self.asfreq.method)

    def test_has_how(self):
        self.assertTrue(hasattr(self.asfreq, 'how'))

    def test_how(self):
        self.assertIsNone(self.asfreq.how)

    def test_has_normalized(self):
        self.assertTrue(hasattr(self.asfreq, 'normalize'))

    def test_normalize(self):
        self.assertIsInstance(self.asfreq.normalize, bool)
        self.assertFalse(self.asfreq.normalize)

    def test_has_fill_value(self):
        self.assertTrue(hasattr(self.asfreq, 'fill_value'))

    def test_fill_value(self):
        self.assertIsNone(self.asfreq.fill_value)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.freq = '1D'
        self.method = 'bfill'
        self.how = 'start'
        self.normalize = True
        self.fill_value = 42.0
        self.asfreq = AsFreq(
            self.freq,
            self.method,
            self.how,
            self.normalize,
            self.fill_value
        )

    def test_method(self):
        self.assertEqual(self.method, self.asfreq.method)

    def test_how(self):
        self.assertEqual(self.how, self.asfreq.how)

    def test_normalize(self):
        self.assertIsInstance(self.asfreq.normalize, bool)
        self.assertTrue(self.asfreq.normalize)

    def test_fill_value(self):
        self.assertEqual(self.fill_value, self.asfreq.fill_value)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.freq = '1D'
        self.method = 'bfill'
        self.how = 'start'
        self.normalize = True
        self.fill_value = 42.0
        self.asfreq = AsFreq(
            self.freq,
            self.method,
            self.how,
            self.normalize,
            self.fill_value
        )

    def test_callable(self):
        self.assertTrue(callable(self.asfreq))

    def test_asfreq_called(self):
        df = pd.DataFrame(range(10))
        df.asfreq = Mock(return_value='expected')
        actual = self.asfreq(df)
        df.asfreq.assert_called_once_with(
            self.freq,
            self.method,
            self.how,
            self.normalize,
            self.fill_value
        )
        self.assertEqual('expected', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        asfreq = AsFreq('1D')
        expected = "AsFreq('1D', None, None, False, None)"
        self.assertEqual(expected, repr(asfreq))

    def test_custom_repr(self):
        asfreq = AsFreq('1D', 'ffill', 'start', True, 42)
        expected = "AsFreq('1D', 'ffill', 'start', True, 42)"
        self.assertEqual(expected, repr(asfreq))

    def test_pickle_works(self):
        asfreq = AsFreq('1D')
        _ = pickle.loads(pickle.dumps(asfreq))


if __name__ == '__main__':
    unittest.main()
