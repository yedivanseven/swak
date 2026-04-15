import pickle
import unittest
from unittest.mock import Mock
from swak.pd import SetIndex


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.keys = ['index']
        self.set = SetIndex(self.keys)

    def test_has_keys(self):
        self.assertTrue(hasattr(self.set, 'keys'))

    def test_keys(self):
        self.assertListEqual(self.keys, self.set.keys)

    def test_has_drop(self):
        self.assertTrue(hasattr(self.set, 'drop'))

    def test_drop(self):
        self.assertIsInstance(self.set.drop, bool)
        self.assertTrue(self.set.drop)

    def test_has_append(self):
        self.assertTrue(hasattr(self.set, 'append'))

    def test_append(self):
        self.assertIsInstance(self.set.append, bool)
        self.assertFalse(self.set.append)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.key = 'index'
        self.keys = 'foo', 'bar'
        self.drop = False
        self.append = True
        self.set = SetIndex(
            self.key,
            *self.keys,
            drop=self.drop,
            append=self.append,
        )

    def test_keys(self):
        self.assertListEqual([self.key, *self.keys], self.set.keys)

    def test_drop(self):
        self.assertIs(self.set.drop, self.drop)

    def test_append(self):
        self.assertIs(self.set.append, self.append)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.keys = 'index'
        self.drop = False
        self.append = True
        self.set = SetIndex(
            self.keys,
            drop=self.drop,
            append=self.append,
        )

    def test_callable(self):
        self.assertTrue(callable(self.set))

    def test_set_index_called(self):
        df = Mock()
        _ = self.set(df)
        df.set_index.assert_called_once_with(
            [self.keys],
            drop=self.drop,
            append=self.append,
            inplace=False
        )

    def test_return_value(self):
        df = Mock()
        df.set_index = Mock(return_value='answer')
        actual = self.set(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.keys = 'index'
        self.drop = False
        self.append = True
        self.set = SetIndex(
            self.keys,
            drop=self.drop,
            append=self.append
        )

    def test_default_repr(self):
        set_index = SetIndex(self.keys)
        expected = "SetIndex(['index'], drop=True, append=False)"
        self.assertEqual(expected, repr(set_index))

    def test_custom_repr(self):
        expected = "SetIndex(['index'], drop=False, append=True)"
        self.assertEqual(expected, repr(self.set))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.set))


if __name__ == '__main__':
    unittest.main()
