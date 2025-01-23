import pickle
import unittest
from logging import Logger
from swak.misc import StdLogger, DEFAULT_FMT, PID_FMT


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = StdLogger('default')

    def test_has_name(self):
        logger = StdLogger('default')
        self.assertTrue(hasattr(logger, 'name'))

    def test_name(self):
        logger = StdLogger('name')
        self.assertEqual('name', logger.name)

    def test_name_stripped(self):
        logger = StdLogger('  name ')
        self.assertEqual('name', logger.name)

    def test_has_level(self):
        logger = StdLogger('default')
        self.assertTrue(hasattr(logger, 'level'))

    def test_level_type(self):
        logger = StdLogger('default')
        self.assertIsInstance(logger.level, int)

    def test_level_value(self):
        logger = StdLogger('default')
        self.assertEqual(10, logger.level)

    def test_has_fmt(self):
        logger = StdLogger('default')
        self.assertTrue(hasattr(logger, 'fmt'))

    def test_fmt_type(self):
        logger = StdLogger('default')
        self.assertIsInstance(logger.fmt, str)

    def test_fmt_value(self):
        logger = StdLogger('default')
        self.assertEqual(DEFAULT_FMT, logger.fmt)

    def test_has_stream(self):
        logger = StdLogger('default')
        self.assertTrue(hasattr(logger, 'stream'))

    def test_stream(self):
        logger = StdLogger('default')
        self.assertEqual('stdout', logger.stream)

    def test_has_logger(self):
        logger = StdLogger('default')
        self.assertTrue(hasattr(logger, 'logger'))

    def test_logger_type(self):
        logger = StdLogger('default')
        self.assertIsInstance(logger.logger, Logger)


class TestCustomAttributes(unittest.TestCase):

    def test_has_level(self):
        logger = StdLogger('default', 20)
        self.assertTrue(hasattr(logger, 'level'))

    def test_level(self):
        obj = object()
        logger = StdLogger('default', obj)
        self.assertIs(logger.level, obj)

    def test_has_fmt(self):
        logger = StdLogger('default', 20, 'format')
        self.assertTrue(hasattr(logger, 'fmt'))

    def test_fmt(self):
        obj = object()
        logger = StdLogger('default', 20, obj)
        self.assertIs(logger.fmt, obj)

    def test_stream(self):
        logger = StdLogger('default', 20, stream='stderr')
        self.assertEqual('stderr', logger.stream)

    def test_stream_stripped(self):
        logger = StdLogger('default', 20, stream='  stderr ')
        self.assertEqual('stderr', logger.stream)

    def test_stream_lowercased(self):
        logger = StdLogger('default', 20, stream='StdErr')
        self.assertEqual('stderr', logger.stream)

    def test_wrong_stream_type_raises(self):
        with self.assertRaises(TypeError):
            _ = StdLogger('default', 20, stream=1)

    def test_wrong_stream_name_raises(self):
        with self.assertRaises(ValueError):
            _ = StdLogger('default', 20, stream='hello world')


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.logger = StdLogger('default')

    def test_has_log(self):
        self.assertTrue(hasattr(self.logger, 'log'))

    def test_callable_log(self):
        self.assertTrue(callable(self.logger.log))

    def test_has_debug(self):
        self.assertTrue(hasattr(self.logger, 'debug'))

    def test_callable_debug(self):
        self.assertTrue(callable(self.logger.debug))

    def test_has_info(self):
        self.assertTrue(hasattr(self.logger, 'info'))

    def test_callable_info(self):
        self.assertTrue(callable(self.logger.info))

    def test_has_warning(self):
        self.assertTrue(hasattr(self.logger, 'warning'))

    def test_callable_warning(self):
        self.assertTrue(callable(self.logger.warning))

    def test_has_error(self):
        self.assertTrue(hasattr(self.logger, 'error'))

    def test_callable_error(self):
        self.assertTrue(callable(self.logger.error))

    def test_has_critical(self):
        self.assertTrue(hasattr(self.logger, 'critical'))

    def test_callable_critical(self):
        self.assertTrue(callable(self.logger.critical))


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.logger = StdLogger('default')

    def test_log_returns_empty_tuple(self):
        with self.assertLogs('default', 10):
            actual = self.logger.log(10, 'msg')
            self.assertTupleEqual((), actual)

    def test_debug_returns_empty_tuple(self):
        with self.assertLogs('default', 10):
            actual = self.logger.debug('msg')
            self.assertTupleEqual((), actual)

    def test_info_returns_empty_tuple(self):
        with self.assertLogs('default', 20):
            actual = self.logger.info('msg')
            self.assertTupleEqual((), actual)

    def test_warning_returns_empty_tuple(self):
        with self.assertLogs('default', 30):
            actual = self.logger.warning('msg')
            self.assertTupleEqual((), actual)

    def test_error_returns_empty_tuple(self):
        with self.assertLogs('default', 40):
            actual = self.logger.error('msg')
            self.assertTupleEqual((), actual)

    def test_critical_returns_empty_tuple(self):
        with self.assertLogs('default', 50):
            actual = self.logger.critical('msg')
            self.assertTupleEqual((), actual)


class TestLogLevel(unittest.TestCase):

    def setUp(self):
        self.logger = StdLogger('default', 30, PID_FMT)

    def test_debug_does_not_log(self):
        with self.assertNoLogs('default', 10):
            _ = self.logger.debug('msg')

    def test_info_does_not_log(self):
        with self.assertNoLogs('default', 20):
            _ = self.logger.info('msg')

    def test_warning_logs(self):
        with self.assertLogs('default', 30):
            _ = self.logger.warning('msg')

    def test_error_logs(self):
        with self.assertLogs('default', 40):
            _ = self.logger.error('msg')

    def test_critical_logs(self):
        with self.assertLogs('default', 50):
            _ = self.logger.critical('msg')


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        logger = StdLogger('default')
        excepted = f"StdLogger('default', 10, '{DEFAULT_FMT}', 'stdout')"
        self.assertEqual(excepted, repr(logger))

    def test_custom_repr(self):
        logger = StdLogger('default', 30, PID_FMT, 'stderr')
        excepted = f"StdLogger('default', 30, '{PID_FMT}', 'stderr')"
        self.assertEqual(excepted, repr(logger))

    def test_pickle_works(self):
        logger = StdLogger('default', 30, PID_FMT, 'stderr')
        _ = pickle.loads(pickle.dumps(logger))


if __name__ == '__main__':
    unittest.main()
