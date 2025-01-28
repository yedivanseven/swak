import pickle
import unittest
from unittest.mock import Mock
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from logging import Logger, getLogger, FileHandler
from swak.funcflow.loggers import PassThroughFileLogger, SHORT_FMT, RAW_FMT


def f(*_):
    return 'msg'


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        with NamedTemporaryFile() as file:
            _ = PassThroughFileLogger(file.name)

    def test_has_file(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'file'))

    def test_file(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertEqual(file.name, logger.file)

    def test_file_stripped(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(f'  {file.name} ')
            self.assertEqual(file.name, logger.file)

    def test_has_level(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'level'))

    def test_level_type(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertIsInstance(logger.level, int)

    def test_level_value(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertEqual(10, logger.level)

    def test_has_fmt(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'fmt'))

    def test_fmt_type(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertIsInstance(logger.fmt, str)

    def test_fmt_value(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertEqual(SHORT_FMT, logger.fmt)

    def test_has_mode(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'mode'))

    def test_mode(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertEqual('a', logger.mode)

    def test_has_encoding(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'encoding'))

    def test_encoding(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertEqual('utf-8', logger.encoding)

    def test_has_delay(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'delay'))

    def test_delay_type(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertIsInstance(logger.delay, bool)

    def test_delay_value(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(logger.delay)

    def test_has_name(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'name'))

    def test_name(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertEqual(file.name, logger.name)

    def test_name_stripped(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(f'  {file.name} ')
            self.assertEqual(file.name, logger.name)

    def test_name_no_dots(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(f'{file.name}.log')
            self.assertEqual(f'{file.name}_log', logger.name)

    def test_has_handler_exists(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'handler_exists'))

    def test_handler_exists_type(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertIsInstance(logger.handler_exists, bool)

    def test_handler_exists_same_logger(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertFalse(logger.handler_exists)
            _ = logger.logger
            self.assertFalse(logger.handler_exists)

    def test_handler_exists_different_logger(self):
        with NamedTemporaryFile() as file:
            existing = getLogger('uix195t')
            handler = FileHandler(file.name)
            existing.addHandler(handler)
            with self.assertRaises(FileExistsError):
                _ = PassThroughFileLogger(file.name).logger

    def test_has_log(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'Log'))

    def test_log_type(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertIsInstance(logger.Log, type)

    def test_has_logger(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'logger'))

    def test_logger_type(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name).logger
            self.assertIsInstance(logger, Logger)

    def test_logger_level(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name).logger
            self.assertEqual(10, logger.level)

    def test_logger_has_handlers(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name).logger
            self.assertTrue(logger.handlers)
            self.assertEqual(1, len(logger.handlers))

    def test_handler_is_file(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertIsInstance(handler, FileHandler)

    def test_handler_file_name(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertEqual(file.name, handler.baseFilename)

    def test_handler_level(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertEqual(10, handler.level)

    def test_handler_mode(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertEqual('a', handler.mode)

    def test_handler_encoding(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertEqual('utf-8', handler.encoding)

    def test_handler_delay(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertTrue(handler.delay)

    def test_has_formatter(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertTrue(hasattr(handler, 'formatter'))

    def test_format_set(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertEqual(SHORT_FMT, handler.formatter._fmt)

    def test_new_logger_same_handler(self):
        with NamedTemporaryFile() as file:
            logger_1 = PassThroughFileLogger(file.name).logger
            handler_1 = logger_1.handlers[0]
            logger_2 = PassThroughFileLogger(file.name).logger
            handler_2 = logger_2.handlers[0]
            self.assertIs(handler_1, handler_2)
            self.assertEqual(1, len(logger_1.handlers))
            self.assertEqual(1, len(logger_2.handlers))


class TestCustomAttributes(unittest.TestCase):

    def test_has_level(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, 20)
            self.assertTrue(hasattr(logger, 'level'))

    def test_level(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, 20)
            self.assertEqual(20, logger.level)
            self.assertEqual(20, logger.logger.level)
            self.assertEqual(20, logger.logger.handlers[0].level)

    def test_has_fmt(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, 20, 'format')
            self.assertTrue(hasattr(logger, 'fmt'))

    def test_fmt(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, 20, RAW_FMT)
            self.assertEqual(RAW_FMT, logger.fmt)
            self.assertEqual(RAW_FMT, logger.logger.handlers[0].formatter._fmt)

    def test_has_mode(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, 20, RAW_FMT, 'w')
            self.assertTrue(hasattr(logger, 'mode'))

    def test_mode(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, 20, RAW_FMT, 'w')
            self.assertEqual('w', logger.mode)
            self.assertEqual('w', logger.logger.handlers[0].mode)

    def test_mode_stripped(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, 20, RAW_FMT, '  w ')
            self.assertEqual('w', logger.mode)
            self.assertEqual('w', logger.logger.handlers[0].mode)

    def test_has_encoding(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(
                file.name,
                20,
                RAW_FMT,
                'w',
                'ascii'
            )
            self.assertTrue(hasattr(logger, 'encoding'))

    def test_encoding(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(
                file.name,
                20,
                RAW_FMT,
                'w',
                'ascii'
            )
            self.assertEqual('ascii', logger.encoding)
            self.assertEqual('ascii', logger.logger.handlers[0].encoding)

    def test_encoding_stripped(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(
                file.name,
                20,
                RAW_FMT,
                'w',
                ' ascii'
            )
            self.assertEqual('ascii', logger.encoding)
            self.assertEqual('ascii', logger.logger.handlers[0].encoding)

    def test_has_delay(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, delay=False)
            self.assertTrue(hasattr(logger, 'delay'))

    def test_delay(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, delay=False)
            self.assertFalse(logger.delay)
            self.assertFalse(logger.logger.handlers[0].delay)

    def test_same_logger_new_level(self):
        with NamedTemporaryFile() as file:
            _ = PassThroughFileLogger(file.name).logger
            logger = PassThroughFileLogger(file.name, 20).logger
            handler = logger.handlers[0]
            self.assertEqual(20, logger.level)
            self.assertEqual(20, handler.level)

    def test_same_logger_new_format(self):
        with NamedTemporaryFile() as file:
            _ = PassThroughFileLogger(file.name).logger
            logger = PassThroughFileLogger(file.name, fmt=RAW_FMT).logger
            handler = logger.handlers[0]
            self.assertEqual(RAW_FMT, handler.formatter._fmt)

    def test_same_logger_new_mode_does_not_change(self):
        with NamedTemporaryFile() as file:
            _ = PassThroughFileLogger(file.name).logger
            logger = PassThroughFileLogger(file.name, mode='w').logger
            handler = logger.handlers[0]
            self.assertEqual('a', handler.mode)

    def test_same_logger_new_encoding_does_not_change(self):
        with NamedTemporaryFile() as file:
            _ = PassThroughFileLogger(file.name).logger
            logger = PassThroughFileLogger(file.name, encoding='ascii').logger
            handler = logger.handlers[0]
            self.assertEqual('utf-8', handler.encoding)

    def test_same_logger_new_delay(self):
        with NamedTemporaryFile() as file:
            _ = PassThroughFileLogger(file.name).logger
            logger = PassThroughFileLogger(file.name, delay=False).logger
            handler = logger.handlers[0]
            self.assertFalse(handler.delay)


class TestMethods(unittest.TestCase):

    def test_has_debug(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'debug'))

    def test_callable_debug(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(callable(logger.debug))

    def test_has_info(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'info'))

    def test_callable_info(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(callable(logger.info))

    def test_has_warning(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'warning'))

    def test_callable_warning(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(callable(logger.warning))

    def test_has_error(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'error'))

    def test_callable_error(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(callable(logger.error))

    def test_has_critical(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(hasattr(logger, 'critical'))

    def test_callable_critical(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            self.assertTrue(callable(logger.critical))


class TestUsage(unittest.TestCase):

    def test_debug_returns_log(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, fmt=RAW_FMT)
            log = logger.debug('msg')
            self.assertIsInstance(log, PassThroughFileLogger.Log)

    def test_info_returns_log(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, fmt=RAW_FMT)
            log = logger.info('msg')
            self.assertIsInstance(log, PassThroughFileLogger.Log)

    def test_warning_returns_log(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, fmt=RAW_FMT)
            log = logger.warning('msg')
            self.assertIsInstance(log, PassThroughFileLogger.Log)

    def test_error_returns_log(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, fmt=RAW_FMT)
            log = logger.error('msg')
            self.assertIsInstance(log, PassThroughFileLogger.Log)

    def test_critical_returns_log(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, fmt=RAW_FMT)
            log = logger.critical('msg')
            self.assertIsInstance(log, PassThroughFileLogger.Log)

    def test_log_callable(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, fmt=RAW_FMT)
            log = logger.debug('msg')
            self.assertTrue(callable(log))

    def test_logs_string_empty(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, fmt=RAW_FMT)
            log = logger.debug('msg')
            out = log()
            self.assertTupleEqual((), out)
            self.assertEqual(b'msg\n', file.read())

    def test_logs_string_arg(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, fmt=RAW_FMT)
            log = logger.debug('msg')
            obj = object()
            out = log(obj)
            self.assertIs(obj, out)
            self.assertEqual(b'msg\n', file.read())

    def test_logs_string_args(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, fmt=RAW_FMT)
            log = logger.debug('msg')
            obj_1 = object()
            obj_2 = object()
            out = log(obj_1, obj_2)
            self.assertTupleEqual((obj_1, obj_2), out)
            self.assertEqual(b'msg\n', file.read())

    def test_logs_call_empty(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, fmt=RAW_FMT)
            mock = Mock(return_value='msg')
            log = logger.debug(mock)
            out = log()
            self.assertTupleEqual((), out)
            mock.assert_called_once_with()
            self.assertEqual(b'msg\n', file.read())

    def test_logs_call_arg(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, fmt=RAW_FMT)
            mock = Mock(return_value='msg')
            log = logger.debug(mock)
            obj = object()
            out = log(obj)
            self.assertIs(obj, out)
            mock.assert_called_once_with(obj)
            self.assertEqual(b'msg\n', file.read())

    def test_logs_call_args(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, fmt=RAW_FMT)
            mock = Mock(return_value='msg')
            log = logger.debug(mock)
            obj_1 = object()
            obj_2 = object()
            out = log(obj_1, obj_2)
            self.assertTupleEqual((obj_1, obj_2), out)
            mock.assert_called_once_with(obj_1, obj_2)
            self.assertEqual(b'msg\n', file.read())

    def test_directory_created(self):
        with TemporaryDirectory() as path:
            file = path + '/subdir/file.log'
            logger = PassThroughFileLogger(file, fmt=RAW_FMT)
            _ = logger.info('msg')()
            with Path(file).open() as file:
                self.assertEqual('msg\n', file.read())


class TestLogLevel(unittest.TestCase):

    def test_debug_does_not_log(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, 30, fmt=RAW_FMT)
            with self.assertNoLogs('name', 10):
                _ = logger.debug('msg')()

    def test_info_does_not_log(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, 30, fmt=RAW_FMT)
            with self.assertNoLogs('name', 20):
                _ = logger.info('msg')()

    def test_warning_logs(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, 30, fmt=RAW_FMT)
            _ = logger.warning('msg')()
            self.assertEqual(b'msg\n', file.read())

    def test_error_logs(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, 30, fmt=RAW_FMT)
            _ = logger.error('msg')()
            self.assertEqual(b'msg\n', file.read())

    def test_critical_logs(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name, 30, fmt=RAW_FMT)
            _ = logger.critical('msg')()
            self.assertEqual(b'msg\n', file.read())


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            excepted = (f"PassThroughFileLogger('{file.name}', 'a', 10,"
                        f" '{SHORT_FMT}', 'utf-8', True)")
            self.assertEqual(excepted, repr(logger))

    def test_custom_repr(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(
                file.name,
                30,
                RAW_FMT,
                'w',
                'ascii',
                False
            )
            excepted = (f"PassThroughFileLogger('{file.name}', 'w', 30,"
                        f" '{RAW_FMT}', 'ascii', False)")
            self.assertEqual(excepted, repr(logger))

    def test_wrapper_pickle_works(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            _ = pickle.loads(pickle.dumps(logger))

    def test_string_pickle_works_before(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            log = logger.debug('msg')
            _ = pickle.loads(pickle.dumps(logger))
            _ = pickle.loads(pickle.dumps(log))

    def test_string_pickle_works_after(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            log = logger.debug('msg')
            _ = log()
            _ = pickle.loads(pickle.dumps(logger))
            _ = pickle.loads(pickle.dumps(log))

    def test_call_pickle_works_before(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            log = logger.debug(f)
            _ = pickle.loads(pickle.dumps(logger))
            _ = pickle.loads(pickle.dumps(log))

    def test_call_pickle_works_after(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            log = logger.debug(f)
            _ = log()
            _ = pickle.loads(pickle.dumps(logger))
            _ = pickle.loads(pickle.dumps(log))

    def test_lambda_pickle_raises(self):
        with NamedTemporaryFile() as file:
            logger = PassThroughFileLogger(file.name)
            log = logger.debug(lambda *_: 'msg')
            with self.assertRaises(AttributeError):
                _ = pickle.dumps(log)


if __name__ == '__main__':
    unittest.main()
