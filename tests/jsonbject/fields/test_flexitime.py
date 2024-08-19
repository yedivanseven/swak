import unittest
import pandas as pd
import datetime as dt
from swak.jsonobject.fields import FlexiTime, FlexiDate


class TestAttributes(unittest.TestCase):

    def test_date_string(self):
        ft = FlexiTime('2021-01-23')
        self.assertEqual('2021-01-23 00:00:00', str(ft))

    def test_time_string(self):
        ft = FlexiTime('2021-01-23 11:24:13.123456')
        self.assertEqual('2021-01-23 11:24:13.123456', str(ft))

    def test_time_t_string(self):
        ft = FlexiTime('2021-01-23T11:24:13.123456')
        self.assertEqual('2021-01-23 11:24:13.123456', str(ft))

    def test_datetime_utc_string(self):
        ft = FlexiTime('2021-01-23 11:24:13.123456Z')
        self.assertEqual('2021-01-23 11:24:13.123456+00:00', str(ft))

    def test_datetime_offset_string(self):
        ft = FlexiTime('2021-01-23 11:24:13.123456+02:00')
        self.assertEqual('2021-01-23 11:24:13.123456+02:00', str(ft))

    def test_datetime_utc_t_string(self):
        ft = FlexiTime('2021-01-23T11:24:13.123456Z')
        self.assertEqual('2021-01-23 11:24:13.123456+00:00', str(ft))

    def test_datetime_offset_t_string(self):
        ft = FlexiTime('2021-01-23T11:24:13.123456+02:00')
        self.assertEqual('2021-01-23 11:24:13.123456+02:00', str(ft))

    def test_date(self):
        ft = FlexiTime(dt.date.fromisoformat('2021-01-23'))
        self.assertEqual('2021-01-23 00:00:00', str(ft))

    def test_datetime(self):
        ft = FlexiTime(dt.datetime.fromisoformat('2021-01-23 12:23:12'))
        self.assertEqual('2021-01-23 12:23:12', str(ft))

    def test_timestamp(self):
        ts = pd.to_datetime('2021-01-23 12:23:12')
        ft = FlexiTime(ts)
        self.assertEqual('2021-01-23 12:23:12', str(ft))

    def test_self(self):
        f = FlexiTime('2021-01-23 12:23:12')
        ft = FlexiTime(f)
        self.assertEqual('2021-01-23 12:23:12', str(ft))

    def test_flexidate(self):
        f = FlexiDate('2021-01-23')
        ft = FlexiTime(f)
        self.assertEqual('2021-01-23 00:00:00', str(ft))


class TestMembers(unittest.TestCase):

    def setUp(self) -> None:
        self.time = '2021-01-23 11:24:13.123456+02:00'
        self.ft = FlexiTime(self.time)

    def test_as_datetime(self):
        self.assertTrue(hasattr(self.ft, 'as_datetime'))
        self.assertIsInstance(self.ft.as_datetime, dt.datetime)
        self.assertEqual(
            dt.datetime.fromisoformat(self.time),
            self.ft.as_datetime
        )

    def test_as_json(self):
        self.assertTrue(hasattr(self.ft, 'as_json'))
        self.assertEqual(self.time, self.ft.as_json)

    def test_as_dtype(self):
        self.assertTrue(hasattr(self.ft, 'as_dtype'))
        self.assertIsInstance(self.ft.as_dtype, pd.Timestamp)
        self.assertEqual(pd.to_datetime(self.time), self.ft.as_dtype)


class TestMagic(unittest.TestCase):

    def setUp(self) -> None:
        self.ft = FlexiTime('2021-01-23 11:24:13')
        self.delta = dt.timedelta(2)

    def test_getattr(self):
        self.assertTrue(hasattr(self.ft, 'fromisoformat'))

    def test_repr(self):
        self.assertEqual("FlexiTime('2021-01-23 11:24:13')", repr(self.ft))

    def test_lt(self):
        self.assertLess(self.ft, '2021-01-23 12:24:13')
        self.assertLess(self.ft, pd.to_datetime('2021-01-23 12:24:13'))
        self.assertLess(self.ft, dt.date.fromisoformat('2021-01-24'))
        self.assertLess(
            self.ft,
            dt.datetime.fromisoformat('2021-01-23 12:24:13')
        )
        self.assertLess(self.ft, FlexiTime('2021-01-23 12:24:13'))
        self.assertLess(self.ft, FlexiDate('2021-01-24'))

    def test_le(self):
        self.assertLessEqual(self.ft, '2021-01-23 12:24:13')
        self.assertLessEqual(self.ft, pd.to_datetime('2021-01-23 12:24:13'))
        self.assertLessEqual(self.ft, dt.date.fromisoformat('2021-01-24'))
        self.assertLessEqual(
            self.ft,
            dt.datetime.fromisoformat('2021-01-23 12:24:13')
        )
        self.assertLessEqual(self.ft, FlexiTime('2021-01-23 12:24:13'))
        self.assertLessEqual(self.ft, FlexiDate('2021-01-24'))

    def test_gt(self):
        self.assertGreater(self.ft, '2021-01-23 10:24:13')
        self.assertGreater(self.ft, pd.to_datetime('2021-01-23 10:24:13'))
        self.assertGreater(self.ft, dt.date.fromisoformat('2021-01-22'))
        self.assertGreater(
            self.ft,
            dt.datetime.fromisoformat('2021-01-23 10:24:13')
        )
        self.assertGreater(self.ft, FlexiTime('2021-01-23 10:24:13'))
        self.assertGreater(self.ft, FlexiDate('2021-01-22'))

    def test_ge(self):
        self.assertGreaterEqual(self.ft, '2021-01-23 10:24:13')
        self.assertGreaterEqual(self.ft, pd.to_datetime('2021-01-23 10:24:13'))
        self.assertGreaterEqual(self.ft, dt.date.fromisoformat('2021-01-22'))
        self.assertGreaterEqual(
            self.ft,
            dt.datetime.fromisoformat('2021-01-23 10:24:13')
        )
        self.assertGreaterEqual(self.ft, FlexiTime('2021-01-23 10:24:13'))
        self.assertGreaterEqual(self.ft, FlexiDate('2021-01-22'))

    def test_eq(self):
        ft = FlexiTime('2021-01-23')
        self.assertEqual(ft, '2021-01-23')
        self.assertEqual(ft, pd.to_datetime('2021-01-23'))
        self.assertEqual(ft, dt.date.fromisoformat('2021-01-23'))
        self.assertEqual(ft, dt.datetime.fromisoformat('2021-01-23'))
        self.assertEqual(ft, FlexiTime('2021-01-23'))
        self.assertEqual(ft, FlexiDate('2021-01-23'))

    def test_ne(self):
        self.assertNotEqual(self.ft, '2021-01-24')
        self.assertNotEqual(self.ft, pd.to_datetime('2021-01-24'))
        self.assertNotEqual(self.ft, dt.date.fromisoformat('2021-01-24'))
        self.assertNotEqual(self.ft, dt.datetime.fromisoformat('2021-01-24'))
        self.assertNotEqual(self.ft, FlexiTime('2021-01-24'))
        self.assertNotEqual(self.ft, FlexiDate('2021-01-23'))

    def test_add(self):
        added = self.ft + self.delta
        self.assertIsInstance(added, FlexiTime)
        self.assertEqual('2021-01-25 11:24:13', str(added))
        added = self.ft + pd.to_timedelta(self.delta)
        self.assertIsInstance(added, FlexiTime)
        self.assertEqual('2021-01-25 11:24:13', str(added))

    def test_radd(self):
        added = self.delta + self.ft
        self.assertIsInstance(added, FlexiTime)
        self.assertEqual('2021-01-25 11:24:13', str(added))
        added = pd.to_timedelta(self.delta) + self.ft
        self.assertIsInstance(added, FlexiTime)
        self.assertEqual('2021-01-25 11:24:13', str(added))

    def test_sub(self):
        added = self.ft - self.delta
        self.assertIsInstance(added, FlexiTime)
        self.assertEqual('2021-01-21 11:24:13', str(added))
        added = self.ft - pd.to_timedelta(self.delta)
        self.assertIsInstance(added, FlexiTime)
        self.assertEqual('2021-01-21 11:24:13', str(added))

    def test_hash(self):
        _ = hash(self.ft)


if __name__ == '__main__':
    unittest.main()
