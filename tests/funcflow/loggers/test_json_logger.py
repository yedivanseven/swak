import sys
import uuid
import unittest
import pickle
from unittest.mock import Mock
from logging import Logger
from swak.misc import JsonStreamHandler
from swak.funcflow.loggers import PassThroughJsonLogger


def f(*_):
    return {'answer': 42}


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = PassThroughJsonLogger('default')

    def test_has_name(self):
        logger = PassThroughJsonLogger('default')
        self.assertTrue(hasattr(logger, 'name'))

    def test_name(self):
        logger = PassThroughJsonLogger('default')
        self.assertEqual('default', logger.name)

    def test_name_stripped(self):
        logger = PassThroughJsonLogger('  name ')
        self.assertEqual('name', logger.name)

    def test_has_level(self):
        logger = PassThroughJsonLogger('default')
        self.assertTrue(hasattr(logger, 'level'))

    def test_level_type(self):
        logger = PassThroughJsonLogger('default')
        self.assertIsInstance(logger.level, int)

    def test_level_value(self):
        logger = PassThroughJsonLogger('default')
        self.assertEqual(10, logger.level)

    def test_has_stream(self):
        logger = PassThroughJsonLogger('default')
        self.assertTrue(hasattr(logger, 'stream'))

    def test_stream(self):
        logger = PassThroughJsonLogger('default')
        self.assertEqual('stdout', logger.stream)

    def test_has_field(self):
        logger = PassThroughJsonLogger('default')
        self.assertTrue(hasattr(logger, 'field'))

    def test_field(self):
        logger = PassThroughJsonLogger('default')
        self.assertTupleEqual(('levelname', 'name'), logger.field)

    def test_has_fields(self):
        logger = PassThroughJsonLogger('default')
        self.assertTrue(hasattr(logger, 'fields'))

    def test_fields(self):
        logger = PassThroughJsonLogger('default')
        self.assertTupleEqual((), logger.fields)

    def test_has_extras(self):
        logger = PassThroughJsonLogger('default')
        self.assertTrue(hasattr(logger, 'extras'))

    def test_extras(self):
        logger = PassThroughJsonLogger('default')
        self.assertDictEqual({}, logger.extras)

    def test_has_log(self):
        logger = PassThroughJsonLogger('default')
        self.assertTrue(hasattr(logger, 'Log'))

    def test_log_type(self):
        logger = PassThroughJsonLogger('default')
        self.assertIsInstance(logger.Log, type)

    def test_has_logger(self):
        logger = PassThroughJsonLogger('default')
        self.assertTrue(hasattr(logger, 'logger'))

    def test_logger_type(self):
        logger = PassThroughJsonLogger('default').logger
        self.assertIsInstance(logger, Logger)

    def test_logger_level(self):
        logger = PassThroughJsonLogger('default').logger
        self.assertEqual(10, logger.level)

    def test_logger_has_handlers(self):
        name = str(uuid.uuid4())
        logger = PassThroughJsonLogger(name).logger
        self.assertTrue(logger.handlers)
        self.assertEqual(1, len(logger.handlers))

    def test_handler_is_stream(self):
        name = str(uuid.uuid4())
        logger = PassThroughJsonLogger(name).logger
        handler = logger.handlers[0]
        self.assertIsInstance(handler, JsonStreamHandler)

    def test_handler_stream_is_stdout(self):
        name = str(uuid.uuid4())
        logger = PassThroughJsonLogger(name).logger
        handler = logger.handlers[0]
        self.assertIs(handler.stream, sys.stdout)

    def test_handler_level(self):
        name = str(uuid.uuid4())
        logger = PassThroughJsonLogger(name).logger
        handler = logger.handlers[0]
        self.assertEqual(10, handler.level)

    def test_handler_fields(self):
        name = str(uuid.uuid4())
        logger = PassThroughJsonLogger(name).logger
        handler = logger.handlers[0]
        self.assertSetEqual({'levelname', 'name'}, handler.fields)

    def test_handler_extras(self):
        name = str(uuid.uuid4())
        logger = PassThroughJsonLogger(name).logger
        handler = logger.handlers[0]
        self.assertDictEqual({}, handler.extras)

    def test_new_logger_same_handler(self):
        name = str(uuid.uuid4())
        wrapper_1 = PassThroughJsonLogger(name)
        wrapper_2 = PassThroughJsonLogger(name)
        self.assertIsNot(wrapper_1, wrapper_2)
        logger_1 = wrapper_1.logger
        logger_2 = wrapper_2.logger
        self.assertIs(logger_1, logger_2)
        self.assertEqual(1, len(logger_1.handlers))
        self.assertEqual(1, len(logger_2.handlers))
        handler_1 = logger_1.handlers[0]
        handler_2 = logger_2.handlers[0]
        self.assertIs(handler_1, handler_2)


class TestCustomAttributes(unittest.TestCase):

    def test_has_level(self):
        logger = PassThroughJsonLogger('default', 20)
        self.assertTrue(hasattr(logger, 'level'))

    def test_level(self):
        logger = PassThroughJsonLogger('default', 20)
        self.assertEqual(20, logger.level)
        self.assertEqual(20, logger.logger.level)
        self.assertEqual(20, logger.logger.handlers[0].level)

    def test_level_truncated_lower(self):
        logger = PassThroughJsonLogger(str(uuid.uuid4()), -20)
        self.assertEqual(10, logger.level)
        self.assertEqual(10, logger.logger.level)
        self.assertEqual(10, logger.logger.handlers[0].level)

    def test_level_truncated_upper(self):
        logger = PassThroughJsonLogger(str(uuid.uuid4()), 70)
        self.assertEqual(50, logger.level)
        self.assertEqual(50, logger.logger.level)
        self.assertEqual(50, logger.logger.handlers[0].level)

    def test_stream(self):
        logger = PassThroughJsonLogger('default', 20, stream='stderr')
        self.assertEqual('stderr', logger.stream)

    def test_single_field(self):
        logger = PassThroughJsonLogger('default', 20, 'stderr', 'name')
        self.assertEqual('name', logger.field)

    def test_multiple_fields(self):
        logger = PassThroughJsonLogger(
            'default',
            20,
            'stderr',
            ['name', 'lineno']
        )
        self.assertListEqual(['name', 'lineno'], logger.field)

    def test_additional_field(self):
        logger = PassThroughJsonLogger(
            'default',
            20,
            'stderr',
            'name',
            'lineno'
        )
        self.assertTupleEqual(('lineno',), logger.fields)

    def test_extras(self):
        logger = PassThroughJsonLogger('default', 20, 'stderr', answer=42)
        self.assertDictEqual({'answer': 42}, logger.extras)

    def test_same_logger_new_level(self):
        name = str(uuid.uuid4())
        logger = PassThroughJsonLogger(name).logger
        _ = PassThroughJsonLogger(name, 20).logger
        self.assertEqual(20, logger.level)
        self.assertEqual(20, logger.handlers[0].level)

    def test_new_stream_new_handler(self):
        name = str(uuid.uuid4())
        logger = PassThroughJsonLogger(name).logger
        self.assertEqual(1, len(logger.handlers))
        _ = PassThroughJsonLogger(name, stream='stderr').logger
        self.assertEqual(2, len(logger.handlers))
        self.assertIs(logger.handlers[0].stream, sys.stdout)
        self.assertIs(logger.handlers[1].stream, sys.stderr)


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.logger = PassThroughJsonLogger('default')

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
        self.logger = PassThroughJsonLogger('default')

    def test_log_returns_log(self):
        call = self.logger.log(10, 'msg')
        self.assertIsInstance(call, PassThroughJsonLogger.Log)

    def test_debug_returns_log(self):
        call = self.logger.debug('msg')
        self.assertIsInstance(call, PassThroughJsonLogger.Log)

    def test_info_returns_log(self):
        call = self.logger.info('msg')
        self.assertIsInstance(call, PassThroughJsonLogger.Log)

    def test_warning_returns_log(self):
        call = self.logger.warning('msg')
        self.assertIsInstance(call, PassThroughJsonLogger.Log)

    def test_error_returns_log(self):
        call = self.logger.error('msg')
        self.assertIsInstance(call, PassThroughJsonLogger.Log)

    def test_critical_returns_log(self):
        call = self.logger.critical('msg')
        self.assertIsInstance(call, PassThroughJsonLogger.Log)

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
        self.logger = PassThroughJsonLogger('default', 30)

    def test_debug_does_not_log(self):
        with self.assertNoLogs('default', 10):
            _ = self.logger.debug('msg')()
            _ = self.logger.log(10, 'msg')()

    def test_info_does_not_log(self):
        with self.assertNoLogs('default', 30):
            _ = self.logger.info('msg')()
            _ = self.logger.log(20, 'msg')()

    def test_warning_logs(self):
        with self.assertLogs('default', 30):
            _ = self.logger.warning('msg')()
            _ = self.logger.log(30, 'msg')()

    def test_error_logs(self):
        with self.assertLogs('default', 40):
            _ = self.logger.error('msg')()
            _ = self.logger.log(40, 'msg')()

    def test_critical_logs(self):
        with self.assertLogs('default', 50):
            _ = self.logger.critical('msg')()
            _ = self.logger.log(50, 'msg')()


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        logger = PassThroughJsonLogger('default')
        excepted = ("PassThroughJsonLogger('default', 10, "
                    "'stdout', ('levelname', 'name'))")
        self.assertEqual(excepted, repr(logger))

    def test_custom_repr(self):
        logger = PassThroughJsonLogger(
            'default',
            30,
            'stderr',
            'asctime',
            'lineno',
            answer=42
        )
        excepted = ("PassThroughJsonLogger('default', 30, 'stderr', "
                    "'asctime', 'lineno', answer=42)")
        self.assertEqual(excepted, repr(logger))

    def test_wrapper_pickle_works(self):
        logger = PassThroughJsonLogger('default')
        _ = pickle.loads(pickle.dumps(logger))

    def test_string_pickle_works_before(self):
        logger = PassThroughJsonLogger('default')
        log = logger.debug('msg')
        _ = pickle.loads(pickle.dumps(logger))
        _ = pickle.loads(pickle.dumps(log))

    def test_string_pickle_works_after(self):
        logger = PassThroughJsonLogger('default')
        log = logger.debug('msg')
        with self.assertLogs('default', 10):
            _ = log()
        _ = pickle.loads(pickle.dumps(logger))
        _ = pickle.loads(pickle.dumps(log))

    def test_call_pickle_works_before(self):
        logger = PassThroughJsonLogger('default')
        log = logger.debug(f)
        _ = pickle.loads(pickle.dumps(logger))
        _ = pickle.loads(pickle.dumps(log))

    def test_call_pickle_works_after(self):
        logger = PassThroughJsonLogger('default')
        log = logger.debug(f)
        with self.assertLogs('default', 10):
            _ = log()
        _ = pickle.loads(pickle.dumps(logger))
        _ = pickle.loads(pickle.dumps(log))

    def test_lambda_pickle_raises(self):
        logger = PassThroughJsonLogger('default')
        log = logger.debug(lambda *_: 'msg')
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(log)


if __name__ == '__main__':
    unittest.main()
