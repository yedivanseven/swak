import pickle
import unittest
from unittest.mock import Mock
from swak.pd import Rename


def f(x):
    return x.upper()


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.rename = Rename()

    def test_has_mapper(self):
        self.assertTrue(hasattr(self.rename, 'mapper'))

    def test_mapper(self):
        self.assertIsNone(self.rename.mapper)

    def test_has_index(self):
        self.assertTrue(hasattr(self.rename, 'index'))

    def test_index(self):
        self.assertIsNone(self.rename.index)

    def test_has_columns(self):
        self.assertTrue(hasattr(self.rename, 'columns'))

    def test_columns(self):
        self.assertIsNone(self.rename.columns)

    def test_has_axis(self):
        self.assertTrue(hasattr(self.rename, 'axis'))

    def test_axis(self):
        self.assertIsInstance(self.rename.axis, int)
        self.assertEqual(1, self.rename.axis)

    def test_has_level(self):
        self.assertTrue(hasattr(self.rename, 'level'))

    def test_level(self):
        self.assertIsNone(self.rename.level)

    def test_has_errors(self):
        self.assertTrue(hasattr(self.rename, 'errors'))

    def test_errors(self):
        self.assertEqual('ignore', self.rename.errors)

    def test_has_resolved(self):
        self.assertTrue(hasattr(self.rename, 'resolved'))

    def test_resolved(self):
        expected = {'columns': None, 'index': None}
        self.assertDictEqual(expected, self.rename.resolved)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mapper = {'old': 'new'}
        self.index = {'foo': 'bar'}
        self.columns = {'answer': 42}
        self.axis = 'index'
        self.level = 'level'
        self.errors = 'raise'
        self.rename = Rename(
            self.mapper,
            self.index,
            self.columns,
            self.axis,
            self.level,
            self.errors
        )

    def test_mapper(self):
        self.assertDictEqual(self.mapper, self.rename.mapper)

    def test_index(self):
        self.assertDictEqual(self.index, self.rename.index)

    def test_columns(self):
        self.assertDictEqual(self.columns, self.rename.columns)

    def test_axis(self):
        self.assertEqual('index', self.rename.axis)

    def test_level(self):
        self.assertEqual('level', self.rename.level)

    def test_errors(self):
        self.assertEqual('raise', self.rename.errors)

    def test_resolved(self):
        self.assertDictEqual({'axis': 'index'}, self.rename.resolved)

    def test_resolved_mapper_none(self):
        rename = Rename(index=self.index, columns=self.columns)
        expected = {'columns': self.columns, 'index': self.index}
        self.assertDictEqual(expected, rename.resolved)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mapper = {'old': 'new'}
        self.index = {'foo': 'bar'}
        self.columns = {'answer': 42}
        self.axis = 'index'
        self.level = 'level'
        self.errors = 'raise'
        self.rename = Rename(
            self.mapper,
            self.index,
            self.columns,
            self.axis,
            self.level,
            self.errors
        )

    def test_callable(self):
        self.assertTrue(callable(self.rename))

    def test_rename_called_mapper_not_none(self):
        df = Mock()
        _ = self.rename(df)
        df.rename.assert_called_once_with(
            self.mapper,
            axis=self.axis,
            level=self.level,
            inplace=False,
            errors=self.errors
        )

    def test_rename_called_mapper_none(self):
        rename = Rename(index=self.index, errors=self.errors)
        df = Mock()
        _ = rename(df)
        df.rename.assert_called_once_with(
            None,
            index=self.index,
            columns=None,
            level=None,
            inplace=False,
            errors=self.errors
        )

    def test_return_value(self):
        df = Mock()
        df.rename = Mock(return_value='answer')
        actual = self.rename(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        rename = Rename()
        expected = ("Rename(None, index=None, columns=None, axis=1, "
                    "level=None, errors='ignore')")
        self.assertEqual(expected, repr(rename))

    def test_custom_repr(self):
        rename = Rename(
            {'old': 'new'},
            {'foo': 'bar'},
            {'answer': 42},
            'index',
            'level',
            'raise'
        )
        expected = ("Rename({'old': 'new'}, index={'foo': 'bar'}, "
                    "columns={'answer': 42}, axis='index', level='level', "
                    "errors='raise')")
        self.assertEqual(expected, repr(rename))

    def test_pickle_works_with_function(self):
        rename = Rename(f)
        _ = pickle.loads(pickle.dumps(rename))

    def test_raises_with_lambda(self):
        rename = Rename(lambda x: x.upper())
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(rename))


if __name__ == '__main__':
    unittest.main()
