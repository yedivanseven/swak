import unittest
import pickle
from unittest.mock import Mock
from logging import Logger
from swak.funcflow.loggers import PassThroughStdOut, DEFAULT_FMT, PID_FMT


def f(*_):
    return 'msg'


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = PassThroughStdOut('default')

    def test_has_name(self):
        logger = PassThroughStdOut('default')
        self.assertTrue(hasattr(logger, 'name'))

    def test_name(self):
        obj = object()
        logger = PassThroughStdOut(obj)
        self.assertIs(logger.name, obj)

    def test_has_level(self):
        logger = PassThroughStdOut('default')
        self.assertTrue(hasattr(logger, 'level'))

    def test_level_type(self):
        logger = PassThroughStdOut('default')
        self.assertIsInstance(logger.level, int)

    def test_level_value(self):
        logger = PassThroughStdOut('default')
        self.assertEqual(10, logger.level)

    def test_has_fmt(self):
        logger = PassThroughStdOut('default')
        self.assertTrue(hasattr(logger, 'fmt'))

    def test_fmt_type(self):
        logger = PassThroughStdOut('default')
        self.assertIsInstance(logger.fmt, str)

    def test_fmt_value(self):
        logger = PassThroughStdOut('default')
        self.assertEqual(DEFAULT_FMT, logger.fmt)

    def test_has_logger(self):
        logger = PassThroughStdOut('default')
        self.assertTrue(hasattr(logger, 'logger'))

    def test_logger_type(self):
        logger = PassThroughStdOut('default')
        self.assertIsInstance(logger.logger, Logger)

    def test_has_log(self):
        logger = PassThroughStdOut('default')
        self.assertTrue(hasattr(logger, 'Log'))

    def test_log_type(self):
        logger = PassThroughStdOut('default')
        self.assertIsInstance(logger.Log, type)


class TestCustomAttributes(unittest.TestCase):

    def test_has_level(self):
        logger = PassThroughStdOut('default', 20)
        self.assertTrue(hasattr(logger, 'level'))

    def test_level(self):
        obj = object()
        logger = PassThroughStdOut('default', obj)
        self.assertIs(logger.level, obj)

    def test_has_fmt(self):
        logger = PassThroughStdOut('default', 20, 'format')
        self.assertTrue(hasattr(logger, 'fmt'))

    def test_fmt(self):
        obj = object()
        logger = PassThroughStdOut('default', 20, obj)
        self.assertIs(logger.fmt, obj)


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.logger = PassThroughStdOut('default')

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
        self.logger = PassThroughStdOut('default')

    def test_debug_returns_log(self):
        call = self.logger.debug('msg')
        self.assertIsInstance(call, PassThroughStdOut.Log)

    def test_info_returns_log(self):
        call = self.logger.info('msg')
        self.assertIsInstance(call, PassThroughStdOut.Log)

    def test_warning_returns_log(self):
        call = self.logger.warning('msg')
        self.assertIsInstance(call, PassThroughStdOut.Log)

    def test_error_returns_log(self):
        call = self.logger.error('msg')
        self.assertIsInstance(call, PassThroughStdOut.Log)

    def test_critical_returns_log(self):
        call = self.logger.critical('msg')
        self.assertIsInstance(call, PassThroughStdOut.Log)

    def test_log_callable(self):
        log = self.logger.debug('msg')
        self.assertTrue(callable(log))

    def test_logs_string_empty(self):
        log = self.logger.debug('msg')
        with self.assertLogs(self.logger.logger, 10) as msg:
            out = log()
        self.assertTupleEqual((), out)
        self.assertListEqual(['DEBUG:default:msg'], msg.output)

    def test_logs_string_arg(self):
        log = self.logger.debug('msg')
        obj = object()
        with self.assertLogs(self.logger.logger, 10) as msg:
            out = log(obj)
        self.assertIs(obj, out)
        self.assertListEqual(['DEBUG:default:msg'], msg.output)

    def test_logs_string_args(self):
        log = self.logger.debug('msg')
        obj_1 = object()
        obj_2 = object()
        with self.assertLogs(self.logger.logger, 10) as msg:
            out = log(obj_1, obj_2)
        self.assertTupleEqual((obj_1, obj_2), out)
        self.assertListEqual(['DEBUG:default:msg'], msg.output)

    def test_logs_call_empty(self):
        mock = Mock(return_value='msg')
        log = self.logger.debug(mock)
        with self.assertLogs(self.logger.logger, 10) as msg:
            out = log()
        self.assertTupleEqual((), out)
        mock.assert_called_once_with()
        self.assertListEqual(['DEBUG:default:msg'], msg.output)

    def test_logs_call_arg(self):
        mock = Mock(return_value='msg')
        log = self.logger.debug(mock)
        obj = object()
        with self.assertLogs(self.logger.logger, 10) as msg:
            out = log(obj)
        self.assertIs(obj, out)
        mock.assert_called_once_with(obj)
        self.assertListEqual(['DEBUG:default:msg'], msg.output)

    def test_logs_call_args(self):
        mock = Mock(return_value='msg')
        log = self.logger.debug(mock)
        obj_1 = object()
        obj_2 = object()
        with self.assertLogs(self.logger.logger, 10) as msg:
            out = log(obj_1, obj_2)
        self.assertTupleEqual((obj_1, obj_2), out)
        mock.assert_called_once_with(obj_1, obj_2)
        self.assertListEqual(['DEBUG:default:msg'], msg.output)


class TestLogLevel(unittest.TestCase):

    def setUp(self):
        self.logger = PassThroughStdOut('default', 30, PID_FMT)

    def test_debug_logs(self):
        with self.assertLogs(self.logger.logger, 10):
            _ = self.logger.debug('msg')()

    def test_debug_does_not_log(self):
        with self.assertNoLogs(self.logger.logger, 30):
            _ = self.logger.debug('msg')()

    def test_info_logs(self):
        with self.assertLogs(self.logger.logger, 20):
            _ = self.logger.info('msg')()

    def test_info_does_not_log(self):
        with self.assertNoLogs(self.logger.logger, 30):
            _ = self.logger.info('msg')()

    def test_warning_logs(self):
        with self.assertLogs(self.logger.logger, 30):
            _ = self.logger.warning('msg')()

    def test_error_logs(self):
        with self.assertLogs(self.logger.logger, 40):
            _ = self.logger.error('msg')()

    def test_critical_logs(self):
        with self.assertLogs(self.logger.logger, 50):
            _ = self.logger.critical('msg')()


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        logger = PassThroughStdOut('default')
        excepted = f"PassThroughStdOut('default', 10, '{DEFAULT_FMT}')"
        self.assertEqual(excepted, repr(logger))

    def test_custom_repr(self):
        logger = PassThroughStdOut('default', 30, PID_FMT)
        excepted = f"PassThroughStdOut('default', 30, '{PID_FMT}')"
        self.assertEqual(excepted, repr(logger))

    def test_wrapper_pickle_works(self):
        logger = PassThroughStdOut('default')
        _ = pickle.dumps(logger)

    def test_string_pickle_works_before(self):
        logger = PassThroughStdOut('default')
        log = logger.debug('msg')
        _ = pickle.dumps(logger)
        _ = pickle.dumps(log)

    def test_string_pickle_works_after(self):
        logger = PassThroughStdOut('default')
        log = logger.debug('msg')
        with self.assertLogs(logger.logger, 10):
            _ = log()
        _ = pickle.dumps(logger)
        _ = pickle.dumps(log)

    def test_call_pickle_works_before(self):
        logger = PassThroughStdOut('default')
        log = logger.debug(f)
        _ = pickle.dumps(logger)
        _ = pickle.dumps(log)

    def test_call_pickle_works_after(self):
        logger = PassThroughStdOut('default')
        log = logger.debug(f)
        with self.assertLogs(logger.logger, 10):
            _ = log()
        _ = pickle.dumps(logger)
        _ = pickle.dumps(log)

    def test_lambda_pickle_raises(self):
        logger = PassThroughStdOut('default')
        log = logger.debug(lambda *_: 'msg')
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(log)


if __name__ == '__main__':
    unittest.main()
