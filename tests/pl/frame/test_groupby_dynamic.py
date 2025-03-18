import pickle
import unittest
from unittest.mock import Mock
import datetime as dt
import polars as pl
from swak.pl import GroupByDynamic


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.index_column = 'date'
        self.every = '1d'
        self.group_by_dynamic = GroupByDynamic(self.index_column, self.every)

    def test_has_index_column(self):
        self.assertTrue(hasattr(self.group_by_dynamic, 'index_column'))

    def test_index_column(self):
        self.assertEqual(self.index_column, self.group_by_dynamic.index_column)

    def test_has_every(self):
        self.assertTrue(hasattr(self.group_by_dynamic, 'every'))

    def test_every(self):
        self.assertEqual(self.every, self.group_by_dynamic.every)

    def test_has_period(self):
        self.assertTrue(hasattr(self.group_by_dynamic, 'period'))

    def test_period(self):
        self.assertIsNone(self.group_by_dynamic.period)

    def test_has_offset(self):
        self.assertTrue(hasattr(self.group_by_dynamic, 'offset'))

    def test_offset(self):
        self.assertIsNone(self.group_by_dynamic.offset)

    def test_has_include_boundaries(self):
        self.assertTrue(hasattr(self.group_by_dynamic, 'include_boundaries'))

    def test_include_boundaries(self):
        self.assertIs(self.group_by_dynamic.include_boundaries, False)

    def test_has_closed(self):
        self.assertTrue(hasattr(self.group_by_dynamic, 'closed'))

    def test_closed(self):
        self.assertEqual('left', self.group_by_dynamic.closed)

    def test_has_label(self):
        self.assertTrue(hasattr(self.group_by_dynamic, 'label'))

    def test_label(self):
        self.assertEqual('left', self.group_by_dynamic.label)

    def test_has_group_by(self):
        self.assertTrue(hasattr(self.group_by_dynamic, 'group_by'))

    def test_group_by(self):
        self.assertIsNone(self.group_by_dynamic.group_by)

    def test_has_start_by(self):
        self.assertTrue(hasattr(self.group_by_dynamic, 'start_by'))

    def test_start_by(self):
        self.assertEqual('window', self.group_by_dynamic.start_by)


class TestCustomAttributes(unittest.TestCase):

    def setUp(self):
        self.index_column = 'timestamp'
        self.every = dt.timedelta(hours=1)
        self.period = '2h'
        self.offset = dt.timedelta(minutes=30)
        self.include_boundaries = True
        self.closed = '  Both '
        self.label = ' Right  '
        self.group_by = 'category'
        self.start_by = '  DataPoint '
        self.group_by_dynamic = GroupByDynamic(
            self.index_column,
            self.every,
            period=self.period,
            offset=self.offset,
            include_boundaries=self.include_boundaries,
            closed=self.closed,
            label=self.label,
            group_by=self.group_by,
            start_by=self.start_by,
        )

    def test_index_column(self):
        self.assertEqual(self.index_column, self.group_by_dynamic.index_column)

    def test_every(self):
        self.assertEqual(self.every, self.group_by_dynamic.every)

    def test_period(self):
        self.assertEqual(self.period, self.group_by_dynamic.period)

    def test_offset(self):
        self.assertEqual(self.offset, self.group_by_dynamic.offset)

    def test_include_boundaries(self):
        self.assertIs(
            self.group_by_dynamic.include_boundaries,
            self.include_boundaries
        )

    def test_closed(self):
        self.assertEqual('both', self.group_by_dynamic.closed)

    def test_label(self):
        self.assertEqual('right', self.group_by_dynamic.label)

    def test_group_by(self):
        self.assertEqual(self.group_by, self.group_by_dynamic.group_by)

    def test_start_by(self):
        self.assertEqual('datapoint', self.group_by_dynamic.start_by)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.index_column = 'date'
        self.every = '1d'
        self.period = '2d'
        self.offset = '12h'
        self.include_boundaries = True
        self.closed = 'both'
        self.label = 'right'
        self.group_by = 'category'
        self.start_by = 'datapoint'
        self.group_by_dynamic = GroupByDynamic(
            self.index_column,
            self.every,
            period=self.period,
            offset=self.offset,
            include_boundaries=self.include_boundaries,
            closed=self.closed,
            label=self.label,
            group_by=self.group_by,
            start_by=self.start_by,
        )

    def test_callable(self):
        self.assertTrue(callable(self.group_by_dynamic))

    def test_group_by_dynamic_called(self):
        df = Mock()
        _ = self.group_by_dynamic(df)
        df.group_by_dynamic.assert_called_once_with(
            self.index_column,
            every=self.every,
            period=self.period,
            offset=self.offset,
            closed=self.closed,
            label=self.label,
            group_by=self.group_by,
            start_by=self.start_by,
        )

    def test_return_value(self):
        df = Mock()
        df.group_by_dynamic = Mock(return_value='dynamic_grouped_result')
        actual = self.group_by_dynamic(df)
        self.assertEqual('dynamic_grouped_result', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        group_by_dynamic = GroupByDynamic('date', '1d')
        expected = ("GroupByDynamic('date', every='1d', period=None, "
                    "offset=None, closed='left', label='left', "
                    "group_by=None, start_by='window')")
        self.assertEqual(expected, repr(group_by_dynamic))

    def test_custom_repr(self):
        group_by_dynamic = GroupByDynamic(
            'timestamp',
            '1h',
            period='2h',
            offset='30m',
            include_boundaries=True,
            closed='both',
            label='right',
            group_by='category',
            start_by='datapoint',
        )
        expected = ("GroupByDynamic('timestamp', every='1h', period='2h', "
                    "offset='30m', closed='both', label='right', "
                    "group_by='category', start_by='datapoint')")
        self.assertEqual(expected, repr(group_by_dynamic))

    def test_pickle_works(self):
        group_by_dynamic = GroupByDynamic(
            pl.col('timestamp'),
            '1h',
            period='2h',
            group_by=pl.col('category')
        )
        _ = pickle.loads(pickle.dumps(group_by_dynamic))


if __name__ == '__main__':
    unittest.main()
