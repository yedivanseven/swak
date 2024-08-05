import unittest
import pandas as pd
import datetime as dt
from swak.jsonobject.fields import FlexiDate, FlexiTime


class TestAttributes(unittest.TestCase):

    def test_date_string(self):
        fd = FlexiDate('2021-01-23')
        self.assertEqual('2021-01-23', str(fd))

    def test_time_string(self):
        fd = FlexiDate('2021-01-23 11:24:13.123456')
        self.assertEqual('2021-01-23', str(fd))

    def test_time_T_string(self):
        fd = FlexiDate('2021-01-23T11:24:13.123456')
        self.assertEqual('2021-01-23', str(fd))

    def test_datetime_utc_string(self):
        fd = FlexiDate('2021-01-23 11:24:13.123456Z')
        self.assertEqual('2021-01-23', str(fd))

    def test_datetime_offset_string(self):
        fd = FlexiDate('2021-01-23 11:24:13.123456+02:00')
        self.assertEqual('2021-01-23', str(fd))

    def test_datetime_utc_T_string(self):
        fd = FlexiDate('2021-01-23T11:24:13.123456Z')
        self.assertEqual('2021-01-23', str(fd))

    def test_datetime_offset_T_string(self):
        fd = FlexiDate('2021-01-23T11:24:13.123456+02:00')
        self.assertEqual('2021-01-23', str(fd))

    def test_date(self):
        fd = FlexiDate(dt.date.fromisoformat('2021-01-23'))
        self.assertEqual('2021-01-23', str(fd))

    def test_datetime(self):
        fd = FlexiDate(dt.datetime.fromisoformat('2021-01-23 12:23:12'))
        self.assertEqual('2021-01-23', str(fd))

    def test_timestamp(self):
        ts = pd.to_datetime('2021-01-23 12:23:12')
        fd = FlexiDate(ts)
        self.assertEqual('2021-01-23', str(fd))

    def test_self(self):
        f = FlexiDate('2021-01-23')
        fd = FlexiDate(f)
        self.assertEqual('2021-01-23', str(fd))

    def test_flexitime(self):
        f = FlexiDate(FlexiTime('2021-01-23'))
        fd = FlexiDate(f)
        self.assertEqual('2021-01-23', str(fd))


class TestMembers(unittest.TestCase):

    def setUp(self) -> None:
        self.date = '2021-01-23'
        self.fd = FlexiDate(self.date)

    def test_as_date(self):
        self.assertTrue(hasattr(self.fd, 'as_date'))
        self.assertIsInstance(self.fd.as_date, dt.date)
        self.assertEqual(dt.date.fromisoformat(self.date), self.fd.as_date)

    def test_as_json(self):
        self.assertTrue(hasattr(self.fd, 'as_json'))
        self.assertIsInstance(self.fd.as_json, str)
        self.assertEqual(self.date, self.fd.as_json)

    def test_as_datetime(self):
        self.assertTrue(hasattr(self.fd, 'as_datetime'))
        self.assertIsInstance(self.fd.as_datetime, dt.datetime)
        expected = dt.datetime.fromisoformat(self.date)
        self.assertEqual(expected, self.fd.as_datetime)

    def test_as_dtype(self):
        self.assertTrue(hasattr(self.fd, 'as_dtype'))
        self.assertIsInstance(self.fd.as_dtype, pd.Timestamp)
        self.assertEqual(pd.to_datetime(self.date), self.fd.as_dtype)


class TestMagic(unittest.TestCase):

    def setUp(self) -> None:
        self.fd = FlexiDate('2021-01-23')
        self.delta = dt.timedelta(2)

    def test_getattr(self):
        self.assertTrue(hasattr(self.fd, 'fromisoformat'))

    def test_repr(self):
        self.assertEqual("FlexiDate('2021-01-23')", repr(self.fd))

    def test_lt(self):
        self.assertLess(self.fd, '2021-01-24')
        self.assertLess(self.fd, pd.to_datetime('2021-01-24'))
        self.assertLess(self.fd, dt.date.fromisoformat('2021-01-24'))
        self.assertLess(self.fd, dt.datetime.fromisoformat('2021-01-24'))
        self.assertLess(self.fd, FlexiDate('2021-01-24'))
        self.assertLess(self.fd, FlexiTime('2021-01-24'))

    def test_le(self):
        self.assertLessEqual(self.fd, '2021-01-24')
        self.assertLessEqual(self.fd, pd.to_datetime('2021-01-24'))
        self.assertLessEqual(self.fd, dt.date.fromisoformat('2021-01-24'))
        self.assertLessEqual(self.fd, dt.datetime.fromisoformat('2021-01-24'))
        self.assertLessEqual(self.fd, FlexiDate('2021-01-24'))
        self.assertLessEqual(self.fd, FlexiTime('2021-01-24'))

    def test_gt(self):
        self.assertGreater(self.fd, '2021-01-22')
        self.assertGreater(self.fd, pd.to_datetime('2021-01-22'))
        self.assertGreater(self.fd, dt.date.fromisoformat('2021-01-22'))
        self.assertGreater(self.fd, dt.datetime.fromisoformat('2021-01-22'))
        self.assertGreater(self.fd, FlexiDate('2021-01-22'))
        self.assertGreater(self.fd, FlexiTime('2021-01-22'))

    def test_ge(self):
        self.assertGreaterEqual(self.fd, '2021-01-23')
        self.assertGreaterEqual(self.fd, pd.to_datetime('2021-01-23'))
        self.assertGreaterEqual(self.fd, dt.date.fromisoformat('2021-01-23'))
        self.assertGreaterEqual(
            self.fd,
            dt.datetime.fromisoformat('2021-01-23')
        )
        self.assertGreaterEqual(self.fd, FlexiDate('2021-01-23'))
        self.assertGreaterEqual(self.fd, FlexiTime('2021-01-23'))

    def test_eq(self):
        self.assertEqual(self.fd, '2021-01-23')
        self.assertEqual(self.fd, pd.to_datetime('2021-01-23'))
        self.assertEqual(self.fd, dt.date.fromisoformat('2021-01-23'))
        self.assertEqual(self.fd, dt.datetime.fromisoformat('2021-01-23'))
        self.assertEqual(self.fd, FlexiDate('2021-01-23'))
        self.assertEqual(self.fd, FlexiTime('2021-01-23'))

    def test_ne(self):
        self.assertNotEqual(self.fd, '2021-01-24')
        self.assertNotEqual(self.fd, pd.to_datetime('2021-01-24'))
        self.assertNotEqual(self.fd, dt.date.fromisoformat('2021-01-24'))
        self.assertNotEqual(self.fd, dt.datetime.fromisoformat('2021-01-24'))
        self.assertNotEqual(self.fd, FlexiDate('2021-01-24'))
        self.assertNotEqual(self.fd, FlexiTime('2021-01-24'))

    def test_add(self):
        added = self.fd + self.delta
        self.assertIsInstance(added, FlexiDate)
        self.assertEqual('2021-01-25', str(added))
        added = self.fd + pd.to_timedelta(self.delta)
        self.assertIsInstance(added, FlexiDate)
        self.assertEqual('2021-01-25', str(added))

    def test_radd(self):
        added = self.delta + self.fd
        self.assertIsInstance(added, FlexiDate)
        self.assertEqual('2021-01-25', str(added))
        added = pd.to_timedelta(self.delta) + self.fd
        self.assertIsInstance(added, FlexiDate)
        self.assertEqual('2021-01-25', str(added))

    def test_sub(self):
        added = self.fd - self.delta
        self.assertIsInstance(added, FlexiDate)
        self.assertEqual('2021-01-21', str(added))
        added = self.fd - pd.to_timedelta(self.delta)
        self.assertIsInstance(added, FlexiDate)
        self.assertEqual('2021-01-21', str(added))

    def test_hash(self):
        _ = hash(self.fd)


if __name__ == '__main__':
    unittest.main()
