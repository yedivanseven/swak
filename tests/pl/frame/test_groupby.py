import pickle
import unittest
from unittest.mock import Mock
import polars as pl
from swak.pl import GroupBy


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.group_by = GroupBy()

    def test_has_by(self):
        self.assertTrue(hasattr(self.group_by, 'by'))

    def test_by(self):
        self.assertTupleEqual((), self.group_by.by)

    def test_has_maintain_order(self):
        self.assertTrue(hasattr(self.group_by, 'maintain_order'))

    def test_maintain_order(self):
        self.assertIs(self.group_by.maintain_order, False)

    def test_has_named_by(self):
        self.assertTrue(hasattr(self.group_by, 'named_by'))

    def test_named_by(self):
        self.assertDictEqual({}, self.group_by.named_by)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.by = 'foo', 'bar'
        self.maintain_order = True
        self.named_by = {'baz': 1, 'answer': 42}
        self.group_by = GroupBy(
            *self.by,
            maintain_order=self.maintain_order,
            **self.named_by
        )

    def test_by(self):
        self.assertTupleEqual(self.by, self.group_by.by)

    def test_maintain_order(self):
        self.assertIs(self.group_by.maintain_order, True)

    def test_named_by(self):
        self.assertDictEqual(self.named_by, self.group_by.named_by)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.by = 'foo', 'bar'
        self.maintain_order = True
        self.named_by = {'baz': 1}
        self.group_by = GroupBy(
            *self.by,
            maintain_order=self.maintain_order,
            **self.named_by
        )

    def test_callable(self):
        self.assertTrue(callable(self.group_by))

    def test_group_by_called(self):
        df = Mock()
        _ = self.group_by(df)
        df.group_by.assert_called_once_with(
            *self.by,
            maintain_order=self.maintain_order,
            **self.named_by
        )

    def test_return_value(self):
        df = Mock()
        df.group_by = Mock(return_value='grouped_result')
        actual = self.group_by(df)
        self.assertEqual('grouped_result', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        group_by = GroupBy()
        expected = 'GroupBy(maintain_order=False)'
        self.assertEqual(expected, repr(group_by))

    def test_custom_repr(self):
        group_by = GroupBy('foo', 'bar', maintain_order=True, answer=42)
        expected = "GroupBy('foo', 'bar', maintain_order=True, answer=42)"
        self.assertEqual(expected, repr(group_by))

    def test_pickle_works(self):
        group_by = GroupBy(
            pl.col('col1'),
            pl.col('col2'),
            maintain_order=True,
            col=pl.col('col3')
        )
        _ = pickle.loads(pickle.dumps(group_by))


if __name__ == '__main__':
    unittest.main()
