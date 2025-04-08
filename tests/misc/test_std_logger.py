import sys
import uuid
import pickle
import unittest
from logging import Logger, StreamHandler
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
        logger = StdLogger('default').logger
        self.assertIsInstance(logger, Logger)

    def test_logger_level(self):
        logger = StdLogger('default').logger
        self.assertEqual(10, logger.level)

    def test_logger_has_handlers(self):
        name = str(uuid.uuid4())
        logger = StdLogger(name).logger
        self.assertTrue(logger.handlers)
        self.assertEqual(1, len(logger.handlers))

    def test_handler_is_stream(self):
        name = str(uuid.uuid4())
        logger = StdLogger(name).logger
        handler = logger.handlers[0]
        self.assertIsInstance(handler, StreamHandler)

    def test_handler_stream_is_stdout(self):
        name = str(uuid.uuid4())
        logger = StdLogger(name).logger
        handler = logger.handlers[0]
        self.assertIs(handler.stream, sys.stdout)

    def test_handler_level(self):
        name = str(uuid.uuid4())
        logger = StdLogger(name).logger
        handler = logger.handlers[0]
        self.assertEqual(10, handler.level)

    def test_new_logger_same_handler(self):
        name = str(uuid.uuid4())
        logger_1 = StdLogger(name).logger
        handler_1 = logger_1.handlers[0]
        logger_2 = StdLogger(name).logger
        handler_2 = logger_2.handlers[0]
        self.assertIs(handler_1, handler_2)
        self.assertEqual(1, len(logger_1.handlers))
        self.assertEqual(1, len(logger_2.handlers))

    def test_has_formatter(self):
        name = str(uuid.uuid4())
        logger = StdLogger(name).logger
        handler = logger.handlers[0]
        self.assertTrue(hasattr(handler, 'formatter'))

    def test_format_set(self):
        name = str(uuid.uuid4())
        logger = StdLogger(name).logger
        handler = logger.handlers[0]
        self.assertEqual(DEFAULT_FMT, handler.formatter._fmt)


class TestCustomAttributes(unittest.TestCase):

    def test_has_level(self):
        logger = StdLogger('default', 20)
        self.assertTrue(hasattr(logger, 'level'))

    def test_level(self):
        logger = StdLogger('default', 20)
        self.assertEqual(logger.level, 20)

    def test_level_truncated_upper(self):
        logger = StdLogger(str(uuid.uuid4()), 70)
        self.assertEqual(50, logger.level)
        actual = logger.logger
        self.assertEqual(50, actual.level)
        handler = actual.handlers[0]
        self.assertEqual(50, handler.level)

    def test_level_truncated_lower(self):
        logger = StdLogger(str(uuid.uuid4()), -20)
        self.assertEqual(10, logger.level)
        actual = logger.logger
        self.assertEqual(10, actual.level)
        handler = actual.handlers[0]
        self.assertEqual(10, handler.level)

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

    def test_same_logger_new_level(self):
        name = str(uuid.uuid4())
        logger = StdLogger(name).logger
        _ = StdLogger(name, 20).logger
        self.assertEqual(20, logger.level)
        self.assertEqual(20, logger.handlers[0].level)

    def test_same_logger_new_format(self):
        name = str(uuid.uuid4())
        logger = StdLogger(name).logger
        _ = StdLogger(name, fmt=PID_FMT).logger
        handler = logger.handlers[0]
        self.assertEqual(PID_FMT, handler.formatter._fmt)

    def test_new_stream_new_handler(self):
        name = str(uuid.uuid4())
        logger = StdLogger(name).logger
        self.assertEqual(1, len(logger.handlers))
        _ = StdLogger(name, stream='stderr').logger
        self.assertEqual(2, len(logger.handlers))
        self.assertIs(logger.handlers[0].stream, sys.stdout)
        self.assertIs(logger.handlers[1].stream, sys.stderr)


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
