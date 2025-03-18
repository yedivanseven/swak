import pickle
import unittest
from unittest.mock import Mock
from swak.pd import GroupBy

def f(x: str) -> str:
    return x.lower()


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.groupby = GroupBy()

    def test_has_by(self):
        self.assertTrue(hasattr(self.groupby, 'by'))

    def test_by(self):
        self.assertIsNone(self.groupby.by)

    def test_has_level(self):
        self.assertTrue(hasattr(self.groupby, 'level'))

    def test_level(self):
        self.assertIsNone(self.groupby.level)

    def test_has_as_index(self):
        self.assertTrue(hasattr(self.groupby, 'as_index'))

    def test_as_index(self):
        self.assertIs(self.groupby.as_index, True)

    def test_has_sort(self):
        self.assertTrue(hasattr(self.groupby, 'sort'))

    def test_sort(self):
        self.assertIs(self.groupby.sort, True)

    def test_has_group_keys(self):
        self.assertTrue(hasattr(self.groupby, 'group_keys'))

    def test_group_keys(self):
        self.assertIs(self.groupby.group_keys, True)

    def test_has_observed(self):
        self.assertTrue(hasattr(self.groupby, 'observed'))

    def test_observed(self):
        self.assertIs(self.groupby.observed, False)

    def test_has_dropna(self):
        self.assertTrue(hasattr(self.groupby, 'dropna'))

    def test_dropna(self):
        self.assertIs(self.groupby.dropna, True)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.by = ['col1', 'col2']
        self.level = 1
        self.as_index = False
        self.sort = False
        self.group_keys = False
        self.observed = True
        self.dropna = False
        self.groupby = GroupBy(
            self.by,
            self.level,
            self.as_index,
            self.sort,
            self.group_keys,
            self.observed,
            self.dropna
        )

    def test_by(self):
        self.assertListEqual(self.by, self.groupby.by)

    def test_level(self):
        self.assertEqual(self.level, self.groupby.level)

    def test_as_index(self):
        self.assertIs(self.groupby.as_index, self.as_index)

    def test_sort(self):
        self.assertIs(self.groupby.sort, self.sort)

    def test_group_keys(self):
        self.assertIs(self.groupby.group_keys, self.group_keys)

    def test_observed(self):
        self.assertIs(self.groupby.observed, self.observed)

    def test_dropna(self):
        self.assertIs(self.groupby.dropna, self.dropna)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.by = ['col1', 'col2']
        self.level = 1
        self.as_index = False
        self.sort = False
        self.group_keys = False
        self.observed = True
        self.dropna = False
        self.groupby = GroupBy(
            self.by,
            self.level,
            self.as_index,
            self.sort,
            self.group_keys,
            self.observed,
            self.dropna
        )

    def test_callable(self):
        self.assertTrue(callable(self.groupby))

    def test_groupby_called(self):
        df = Mock()
        _ = self.groupby(df)
        df.groupby.assert_called_once_with(
            self.by,
            0,
            self.level,
            self.as_index,
            self.sort,
            self.group_keys,
            self.observed,
            self.dropna
        )

    def test_return_value(self):
        df = Mock()
        df.groupby = Mock(return_value='answer')
        actual = self.groupby(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        groupby = GroupBy()
        expected = ("GroupBy(None, None, as_index=True, sort=True, "
                    "group_keys=True, observed=False, dropna=True)")
        self.assertEqual(expected, repr(groupby))

    def test_custom_repr(self):
        groupby = GroupBy(
            ['col1', 'col2'],
            1,
            False,
            False,
            False,
            True,
            False
        )
        expected = (
            "GroupBy(['col1', 'col2'], 1, as_index=False, "
            "sort=False, group_keys=False, observed=True, dropna=False)"
        )
        self.assertEqual(expected, repr(groupby))

    def test_pickle_works_with_function(self):
        groupby = GroupBy(f)
        _ = pickle.loads(pickle.dumps(groupby))

    def test_raises_with_lambda(self):
        groupby = GroupBy(lambda x: x.lower())
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(groupby))


if __name__ == '__main__':
    unittest.main()
