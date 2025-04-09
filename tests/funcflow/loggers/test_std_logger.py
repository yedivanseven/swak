import sys
import uuid
import unittest
import pickle
from unittest.mock import Mock
from logging import Logger, StreamHandler
from swak.funcflow.loggers import PassThroughStdLogger, DEFAULT_FMT, PID_FMT


def f(*_):
    return 'msg'


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = PassThroughStdLogger('default')

    def test_has_name(self):
        logger = PassThroughStdLogger('default')
        self.assertTrue(hasattr(logger, 'name'))

    def test_name(self):
        logger = PassThroughStdLogger('default')
        self.assertEqual('default', logger.name)

    def test_name_stripped(self):
        logger = PassThroughStdLogger('  name ')
        self.assertEqual('name', logger.name)

    def test_has_level(self):
        logger = PassThroughStdLogger('default')
        self.assertTrue(hasattr(logger, 'level'))

    def test_level_type(self):
        logger = PassThroughStdLogger('default')
        self.assertIsInstance(logger.level, int)

    def test_level_value(self):
        logger = PassThroughStdLogger('default')
        self.assertEqual(10, logger.level)

    def test_has_fmt(self):
        logger = PassThroughStdLogger('default')
        self.assertTrue(hasattr(logger, 'fmt'))

    def test_fmt_type(self):
        logger = PassThroughStdLogger('default')
        self.assertIsInstance(logger.fmt, str)

    def test_fmt_value(self):
        logger = PassThroughStdLogger('default')
        self.assertEqual(DEFAULT_FMT, logger.fmt)

    def test_has_stream(self):
        logger = PassThroughStdLogger('default')
        self.assertTrue(hasattr(logger, 'stream'))

    def test_stream(self):
        logger = PassThroughStdLogger('default')
        self.assertEqual('stdout', logger.stream)

    def test_has_log(self):
        logger = PassThroughStdLogger('default')
        self.assertTrue(hasattr(logger, 'Log'))

    def test_log_type(self):
        logger = PassThroughStdLogger('default')
        self.assertIsInstance(logger.Log, type)

    def test_has_logger(self):
        logger = PassThroughStdLogger('default')
        self.assertTrue(hasattr(logger, 'logger'))

    def test_logger_type(self):
        logger = PassThroughStdLogger('default').logger
        self.assertIsInstance(logger, Logger)

    def test_logger_level(self):
        logger = PassThroughStdLogger('default').logger
        self.assertEqual(10, logger.level)

    def test_logger_has_handlers(self):
        name = str(uuid.uuid4())
        logger = PassThroughStdLogger(name).logger
        self.assertTrue(logger.handlers)
        self.assertEqual(1, len(logger.handlers))

    def test_handler_is_stream(self):
        name = str(uuid.uuid4())
        logger = PassThroughStdLogger(name).logger
        handler = logger.handlers[0]
        self.assertIsInstance(handler, StreamHandler)

    def test_handler_stream_is_stdout(self):
        name = str(uuid.uuid4())
        logger = PassThroughStdLogger(name).logger
        handler = logger.handlers[0]
        self.assertIs(handler.stream, sys.stdout)

    def test_handler_level(self):
        name = str(uuid.uuid4())
        logger = PassThroughStdLogger(name).logger
        handler = logger.handlers[0]
        self.assertEqual(10, handler.level)

    def test_new_logger_same_handler(self):
        name = str(uuid.uuid4())
        logger_1 = PassThroughStdLogger(name).logger
        handler_1 = logger_1.handlers[0]
        logger_2 = PassThroughStdLogger(name).logger
        handler_2 = logger_2.handlers[0]
        self.assertIs(handler_1, handler_2)
        self.assertEqual(1, len(logger_1.handlers))
        self.assertEqual(1, len(logger_2.handlers))

    def test_has_formatter(self):
        name = str(uuid.uuid4())
        logger = PassThroughStdLogger(name).logger
        handler = logger.handlers[0]
        self.assertTrue(hasattr(handler, 'formatter'))

    def test_format_set(self):
        name = str(uuid.uuid4())
        logger = PassThroughStdLogger(name).logger
        handler = logger.handlers[0]
        self.assertEqual(DEFAULT_FMT, handler.formatter._fmt)


class TestCustomAttributes(unittest.TestCase):

    def test_has_level(self):
        logger = PassThroughStdLogger('default', 20)
        self.assertTrue(hasattr(logger, 'level'))

    def test_level(self):
        logger = PassThroughStdLogger('default', 20)
        self.assertEqual(20, logger.level)
        self.assertEqual(20, logger.logger.level)
        self.assertEqual(20, logger.logger.handlers[0].level)

    def test_level_truncated_lower(self):
        logger = PassThroughStdLogger(str(uuid.uuid4()), -20)
        self.assertEqual(10, logger.level)
        self.assertEqual(10, logger.logger.level)
        self.assertEqual(10, logger.logger.handlers[0].level)

    def test_level_truncated_upper(self):
        logger = PassThroughStdLogger(str(uuid.uuid4()), 70)
        self.assertEqual(50, logger.level)
        self.assertEqual(50, logger.logger.level)
        self.assertEqual(50, logger.logger.handlers[0].level)

    def test_has_fmt(self):
        logger = PassThroughStdLogger('default', 20, 'format')
        self.assertTrue(hasattr(logger, 'fmt'))

    def test_fmt(self):
        logger = PassThroughStdLogger('default', 20, 'format')
        self.assertEqual('format', logger.fmt)

    def test_stream(self):
        logger = PassThroughStdLogger('default', 20, stream='stderr')
        self.assertEqual('stderr', logger.stream)

    def test_stream_stripped(self):
        logger = PassThroughStdLogger('default', 20, stream='  stderr ')
        self.assertEqual('stderr', logger.stream)

    def test_stream_lowercased(self):
        logger = PassThroughStdLogger('default', 20, stream='StdErr')
        self.assertEqual('stderr', logger.stream)

    def test_wrong_stream_type_raises(self):
        with self.assertRaises(TypeError):
            _ = PassThroughStdLogger('default', 20, stream=1)

    def test_wrong_stream_name_raises(self):
        with self.assertRaises(ValueError):
            _ = PassThroughStdLogger('default', 20, stream='hello world')

    def test_same_logger_new_level(self):
        name = str(uuid.uuid4())
        logger = PassThroughStdLogger(name).logger
        _ = PassThroughStdLogger(name, 20).logger
        self.assertEqual(20, logger.level)
        self.assertEqual(20, logger.handlers[0].level)

    def test_same_logger_new_format(self):
        name = str(uuid.uuid4())
        logger = PassThroughStdLogger(name).logger
        _ = PassThroughStdLogger(name, fmt=PID_FMT).logger
        handler = logger.handlers[0]
        self.assertEqual(PID_FMT, handler.formatter._fmt)

    def test_new_stream_new_handler(self):
        name = str(uuid.uuid4())
        logger = PassThroughStdLogger(name).logger
        self.assertEqual(1, len(logger.handlers))
        _ = PassThroughStdLogger(name, stream='stderr').logger
        self.assertEqual(2, len(logger.handlers))
        self.assertIs(logger.handlers[0].stream, sys.stdout)
        self.assertIs(logger.handlers[1].stream, sys.stderr)


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.logger = PassThroughStdLogger('default')

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
        self.logger = PassThroughStdLogger('default')

    def test_debug_returns_log(self):
        call = self.logger.debug('msg')
        self.assertIsInstance(call, PassThroughStdLogger.Log)

    def test_info_returns_log(self):
        call = self.logger.info('msg')
        self.assertIsInstance(call, PassThroughStdLogger.Log)

    def test_warning_returns_log(self):
        call = self.logger.warning('msg')
        self.assertIsInstance(call, PassThroughStdLogger.Log)

    def test_error_returns_log(self):
        call = self.logger.error('msg')
        self.assertIsInstance(call, PassThroughStdLogger.Log)

    def test_critical_returns_log(self):
        call = self.logger.critical('msg')
        self.assertIsInstance(call, PassThroughStdLogger.Log)

    def test_log_callable(self):
        log = self.logger.debug('msg')
        self.assertTrue(callable(log))

    def test_logs_string_empty(self):
        log = self.logger.debug('msg')
        with self.assertLogs('default', 10) as msg:
            out = log()
        self.assertTupleEqual((), out)
        self.assertListEqual(['DEBUG:default:msg'], msg.output)

    def test_logs_string_arg(self):
        log = self.logger.debug('msg')
        obj = object()
        with self.assertLogs('default', 10) as msg:
            out = log(obj)
        self.assertIs(obj, out)
        self.assertListEqual(['DEBUG:default:msg'], msg.output)

    def test_logs_string_args(self):
        log = self.logger.debug('msg')
        obj_1 = object()
        obj_2 = object()
        with self.assertLogs('default', 10) as msg:
            out = log(obj_1, obj_2)
        self.assertTupleEqual((obj_1, obj_2), out)
        self.assertListEqual(['DEBUG:default:msg'], msg.output)

    def test_logs_call_empty(self):
        mock = Mock(return_value='msg')
        log = self.logger.debug(mock)
        with self.assertLogs('default', 10) as msg:
            out = log()
        self.assertTupleEqual((), out)
        mock.assert_called_once_with()
        self.assertListEqual(['DEBUG:default:msg'], msg.output)

    def test_logs_call_arg(self):
        mock = Mock(return_value='msg')
        log = self.logger.debug(mock)
        obj = object()
        with self.assertLogs('default', 10) as msg:
            out = log(obj)
        self.assertIs(obj, out)
        mock.assert_called_once_with(obj)
        self.assertListEqual(['DEBUG:default:msg'], msg.output)

    def test_logs_call_args(self):
        mock = Mock(return_value='msg')
        log = self.logger.debug(mock)
        obj_1 = object()
        obj_2 = object()
        with self.assertLogs('default', 10) as msg:
            out = log(obj_1, obj_2)
        self.assertTupleEqual((obj_1, obj_2), out)
        mock.assert_called_once_with(obj_1, obj_2)
        self.assertListEqual(['DEBUG:default:msg'], msg.output)


class TestLogLevel(unittest.TestCase):

    def setUp(self):
        self.logger = PassThroughStdLogger('default', 30, PID_FMT)

    def test_debug_does_not_log(self):
        with self.assertNoLogs('default', 10):
            _ = self.logger.debug('msg')()

    def test_info_does_not_log(self):
        with self.assertNoLogs('default', 30):
            _ = self.logger.info('msg')()

    def test_warning_logs(self):
        with self.assertLogs('default', 30):
            _ = self.logger.warning('msg')()

    def test_error_logs(self):
        with self.assertLogs('default', 40):
            _ = self.logger.error('msg')()

    def test_critical_logs(self):
        with self.assertLogs('default', 50):
            _ = self.logger.critical('msg')()


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        logger = PassThroughStdLogger('default')
        excepted = (f"PassThroughStdLogger('default', "
                    f"10, '{DEFAULT_FMT}', 'stdout')")
        self.assertEqual(excepted, repr(logger))

    def test_custom_repr(self):
        logger = PassThroughStdLogger('default', 30, PID_FMT, 'stderr')
        excepted = (f"PassThroughStdLogger('default', "
                    f"30, '{PID_FMT}', 'stderr')")
        self.assertEqual(excepted, repr(logger))

    def test_wrapper_pickle_works(self):
        logger = PassThroughStdLogger('default')
        _ = pickle.loads(pickle.dumps(logger))

    def test_string_pickle_works_before(self):
        logger = PassThroughStdLogger('default')
        log = logger.debug('msg')
        _ = pickle.loads(pickle.dumps(logger))
        _ = pickle.loads(pickle.dumps(log))

    def test_string_pickle_works_after(self):
        logger = PassThroughStdLogger('default')
        log = logger.debug('msg')
        with self.assertLogs('default', 10):
            _ = log()
        _ = pickle.loads(pickle.dumps(logger))
        _ = pickle.loads(pickle.dumps(log))

    def test_call_pickle_works_before(self):
        logger = PassThroughStdLogger('default')
        log = logger.debug(f)
        _ = pickle.loads(pickle.dumps(logger))
        _ = pickle.loads(pickle.dumps(log))

    def test_call_pickle_works_after(self):
        logger = PassThroughStdLogger('default')
        log = logger.debug(f)
        with self.assertLogs('default', 10):
            _ = log()
        _ = pickle.loads(pickle.dumps(logger))
        _ = pickle.loads(pickle.dumps(log))

    def test_lambda_pickle_raises(self):
        logger = PassThroughStdLogger('default')
        log = logger.debug(lambda *_: 'msg')
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(log)


if __name__ == '__main__':
    unittest.main()
