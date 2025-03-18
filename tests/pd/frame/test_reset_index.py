import pickle
import unittest
from unittest.mock import Mock
from swak.pd import ResetIndex


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.reset = ResetIndex()

    def test_has_level(self):
        self.assertTrue(hasattr(self.reset, 'level'))

    def test_level(self):
        self.assertIsNone(self.reset.level)

    def test_has_drop(self):
        self.assertTrue(hasattr(self.reset, 'drop'))

    def test_drop(self):
        self.assertIsInstance(self.reset.drop, bool)
        self.assertFalse(self.reset.drop)

    def test_has_col_level(self):
        self.assertTrue(hasattr(self.reset, 'col_level'))

    def test_col_level(self):
        self.assertIsInstance(self.reset.col_level, int)
        self.assertEqual(0, self.reset.col_level)

    def test_has_col_fill(self):
        self.assertTrue(hasattr(self.reset, 'col_fill'))

    def test_col_fill(self):
        self.assertEqual('', self.reset.col_fill)

    def test_has_allow_duplicates(self):
        self.assertTrue(hasattr(self.reset, 'allow_duplicates'))

    def test_allow_duplicates(self):
        self.assertIsInstance(self.reset.allow_duplicates, bool)
        self.assertFalse(self.reset.allow_duplicates)

    def test_has_names(self):
        self.assertTrue(hasattr(self.reset, 'names'))

    def test_names(self):
        self.assertIsNone(self.reset.names)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.level = 'level'
        self.drop = True
        self.col_level = 1
        self.col_fill = 'fill'
        self.allow_duplicates = True
        self.names = 'names'
        self.reset = ResetIndex(
            self.level,
            self.drop,
            self.col_level,
            self.col_fill,
            self.allow_duplicates,
            self.names
        )

    def test_level(self):
        self.assertEqual(self.level, self.reset.level)

    def test_drop(self):
        self.assertIs(self.reset.drop, self.drop)

    def test_col_level(self):
        self.assertEqual(1, self.reset.col_level)

    def test_col_fill(self):
        self.assertEqual('fill', self.reset.col_fill)

    def test_allow_duplicates(self):
        self.assertIs(self.reset.allow_duplicates, self.allow_duplicates)

    def test_names(self):
        self.assertEqual(self.names, self.reset.names)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.level = 'level'
        self.drop = True
        self.col_level = 1
        self.col_fill = 'fill'
        self.allow_duplicates = True
        self.names = 'names'
        self.reset = ResetIndex(
            self.level,
            self.drop,
            self.col_level,
            self.col_fill,
            self.allow_duplicates,
            self.names,
        )

    def test_callable(self):
        self.assertTrue(callable(self.reset))

    def test_reset_index_called(self):
        df = Mock()
        _ = self.reset(df)
        df.reset_index.assert_called_once_with(
            self.level,
            drop=self.drop,
            inplace=False,
            col_level=self.col_level,
            col_fill=self.col_fill,
            allow_duplicates=self.allow_duplicates,
            names=self.names
        )

    def test_return_value(self):
        df = Mock()
        df.reset_index = Mock(return_value='answer')
        actual = self.reset(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        reset = ResetIndex()
        expected = (
            "ResetIndex(None, drop=False, col_level=0, col_fill='',"
            ' allow_duplicates=False, names=None)'
        )
        self.assertEqual(expected, repr(reset))

    def test_custom_repr(self):
        reset = ResetIndex(
            1,
            True,
            2,
            'fill',
            True,
            'names',
        )
        expected = ("ResetIndex(1, drop=True, col_level=2, col_fill='fill', "
                    "allow_duplicates=True, names='names')")
        self.assertEqual(expected, repr(reset))

    def test_pickle_works(self):
        reset = ResetIndex(
            1,
            True,
            2,
            'fill',
            True,
            'names',
        )
        _ = pickle.loads(pickle.dumps(reset))


if __name__ == '__main__':
    unittest.main()
