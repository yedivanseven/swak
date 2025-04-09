import pickle
import unittest
from tempfile import NamedTemporaryFile, TemporaryDirectory
from pathlib import Path
from logging import Logger, getLogger, FileHandler
from swak.misc import FileLogger, PID_FMT, RAW_FMT, SHORT_FMT


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        with NamedTemporaryFile() as file:
            _ = FileLogger(file.name)

    def test_has_file(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'file'))

    def test_file(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertEqual(file.name, logger.file)

    def test_file_stripped(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(f'  {file.name} ')
            self.assertEqual(file.name, logger.file)

    def test_has_level(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'level'))

    def test_level_type(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertIsInstance(logger.level, int)

    def test_level_value(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertEqual(10, logger.level)

    def test_has_fmt(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'fmt'))

    def test_fmt_type(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertIsInstance(logger.fmt, str)

    def test_fmt_value(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertEqual(SHORT_FMT, logger.fmt)

    def test_has_mode(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'mode'))

    def test_mode(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertEqual('a', logger.mode)

    def test_has_encoding(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'encoding'))

    def test_encoding(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertEqual('utf-8', logger.encoding)

    def test_has_delay(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'delay'))

    def test_delay_type(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertIsInstance(logger.delay, bool)

    def test_delay_value(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(logger.delay)

    def test_has_name(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'name'))

    def test_name(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertEqual(file.name, logger.name)

    def test_name_stripped(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(f'  {file.name} ')
            self.assertEqual(file.name, logger.name)

    def test_name_no_dots(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(f'{file.name}.log')
            self.assertEqual(f'{file.name}_log', logger.name)

    def test_has_handler_exists(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'handler_exists'))

    def test_handler_exists_type(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertIsInstance(logger.handler_exists, bool)

    def test_handler_exists_same_logger(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertFalse(logger.handler_exists)
            _ = logger.logger
            self.assertFalse(logger.handler_exists)

    def test_handler_exists_different_logger(self):
        with NamedTemporaryFile() as file:
            existing = getLogger('uix195t')
            handler = FileHandler(file.name)
            existing.addHandler(handler)
            with self.assertRaises(FileExistsError):
                _ = FileLogger(file.name).logger

    def test_has_logger(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'logger'))

    def test_logger_type(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name).logger
            self.assertIsInstance(logger, Logger)

    def test_logger_level(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name).logger
            self.assertEqual(10, logger.level)

    def test_logger_has_handlers(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name).logger
            self.assertTrue(logger.handlers)
            self.assertEqual(1, len(logger.handlers))

    def test_handler_is_file(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertIsInstance(handler, FileHandler)

    def test_handler_file_name(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertEqual(file.name, handler.baseFilename)

    def test_handler_level(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertEqual(10, handler.level)

    def test_handler_mode(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertEqual('a', handler.mode)

    def test_handler_encoding(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertEqual('utf-8', handler.encoding)

    def test_handler_delay(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertTrue(handler.delay)

    def test_has_formatter(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertTrue(hasattr(handler, 'formatter'))

    def test_format_set(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name).logger
            handler = logger.handlers[0]
            self.assertEqual(SHORT_FMT, handler.formatter._fmt)

    def test_new_logger_same_handler(self):
        with NamedTemporaryFile() as file:
            logger_1 = FileLogger(file.name).logger
            handler_1 = logger_1.handlers[0]
            logger_2 = FileLogger(file.name).logger
            handler_2 = logger_2.handlers[0]
            self.assertIs(handler_1, handler_2)
            self.assertEqual(1, len(logger_1.handlers))
            self.assertEqual(1, len(logger_2.handlers))


class TestCustomAttributes(unittest.TestCase):

    def test_has_level(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 20)
            self.assertTrue(hasattr(logger, 'level'))

    def test_level(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 20)
            self.assertEqual(20, logger.level)
            self.assertEqual(20, logger.logger.level)
            self.assertEqual(20, logger.logger.handlers[0].level)

    def test_level_truncated_lower(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, -20)
            self.assertEqual(10, logger.level)
            self.assertEqual(10, logger.logger.level)
            self.assertEqual(10, logger.logger.handlers[0].level)

    def test_level_truncated_upper(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 70)
            self.assertEqual(50, logger.level)
            self.assertEqual(50, logger.logger.level)
            self.assertEqual(50, logger.logger.handlers[0].level)

    def test_has_fmt(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 20, 'format')
            self.assertTrue(hasattr(logger, 'fmt'))

    def test_fmt(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 20, PID_FMT)
            self.assertEqual(PID_FMT, logger.fmt)
            self.assertEqual(PID_FMT, logger.logger.handlers[0].formatter._fmt)

    def test_has_mode(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 20, PID_FMT, 'w')
            self.assertTrue(hasattr(logger, 'mode'))

    def test_mode(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 20, PID_FMT, 'w')
            self.assertEqual('w', logger.mode)
            self.assertEqual('w', logger.logger.handlers[0].mode)

    def test_mode_stripped(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 20, PID_FMT, '  w ')
            self.assertEqual('w', logger.mode)
            self.assertEqual('w', logger.logger.handlers[0].mode)

    def test_has_encoding(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 20, PID_FMT, 'w', 'ascii')
            self.assertTrue(hasattr(logger, 'encoding'))

    def test_encoding(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 20, PID_FMT, 'w', 'ascii')
            self.assertEqual('ascii', logger.encoding)
            self.assertEqual('ascii', logger.logger.handlers[0].encoding)

    def test_encoding_stripped(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 20, PID_FMT, 'w', ' ascii')
            self.assertEqual('ascii', logger.encoding)
            self.assertEqual('ascii', logger.logger.handlers[0].encoding)

    def test_has_delay(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, delay=False)
            self.assertTrue(hasattr(logger, 'delay'))

    def test_delay(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, delay=False)
            self.assertFalse(logger.delay)
            self.assertFalse(logger.logger.handlers[0].delay)

    def test_same_logger_new_level(self):
        with NamedTemporaryFile() as file:
            _ = FileLogger(file.name).logger
            logger = FileLogger(file.name, 20).logger
            handler = logger.handlers[0]
            self.assertEqual(20, logger.level)
            self.assertEqual(20, handler.level)

    def test_same_logger_new_format(self):
        with NamedTemporaryFile() as file:
            _ = FileLogger(file.name).logger
            logger = FileLogger(file.name, fmt=PID_FMT).logger
            handler = logger.handlers[0]
            self.assertEqual(PID_FMT, handler.formatter._fmt)

    def test_same_logger_new_mode_does_not_change(self):
        with NamedTemporaryFile() as file:
            _ = FileLogger(file.name).logger
            logger = FileLogger(file.name, mode='w').logger
            handler = logger.handlers[0]
            self.assertEqual('a', handler.mode)

    def test_same_logger_new_encoding_does_not_change(self):
        with NamedTemporaryFile() as file:
            _ = FileLogger(file.name).logger
            logger = FileLogger(file.name, encoding='ascii').logger
            handler = logger.handlers[0]
            self.assertEqual('utf-8', handler.encoding)

    def test_same_logger_new_delay(self):
        with NamedTemporaryFile() as file:
            _ = FileLogger(file.name).logger
            logger = FileLogger(file.name, delay=False).logger
            handler = logger.handlers[0]
            self.assertFalse(handler.delay)


class TestMethods(unittest.TestCase):

    def test_has_log(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'log'))

    def test_callable_log(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(callable(logger.log))

    def test_has_debug(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'debug'))

    def test_callable_debug(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(callable(logger.debug))

    def test_has_info(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'info'))

    def test_callable_info(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(callable(logger.info))

    def test_has_warning(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'warning'))

    def test_callable_warning(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(callable(logger.warning))

    def test_has_error(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'error'))

    def test_callable_error(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(callable(logger.error))

    def test_has_critical(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(hasattr(logger, 'critical'))

    def test_callable_critical(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            self.assertTrue(callable(logger.critical))


class TestUsage(unittest.TestCase):

    def test_log(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, fmt=RAW_FMT)
            actual = logger.log(10, 'msg')
            self.assertTupleEqual((), actual)
            self.assertEqual(b'msg\n', file.read())

    def test_debug(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, fmt=RAW_FMT)
            actual = logger.debug('msg')
            self.assertTupleEqual((), actual)
            self.assertEqual(b'msg\n', file.read())

    def test_info(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, fmt=RAW_FMT)
            actual = logger.info('msg')
            self.assertTupleEqual((), actual)
            self.assertEqual(b'msg\n', file.read())

    def test_warning(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, fmt=RAW_FMT)
            actual = logger.warning('msg')
            self.assertTupleEqual((), actual)
            self.assertEqual(b'msg\n', file.read())

    def test_error(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, fmt=RAW_FMT)
            actual = logger.error('msg')
            self.assertTupleEqual((), actual)
            self.assertEqual(b'msg\n', file.read())

    def test_critical(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, fmt=RAW_FMT)
            actual = logger.critical('msg')
            self.assertTupleEqual((), actual)
            self.assertEqual(b'msg\n', file.read())

    def test_directory_created(self):
        with TemporaryDirectory() as path:
            file = path + '/subdir/file.log'
            logger = FileLogger(file, fmt=RAW_FMT)
            _ = logger.info('msg')
            with Path(file).open() as file:
                self.assertEqual('msg\n', file.read())


class TestLogLevel(unittest.TestCase):

    def test_debug_does_not_log(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 30, fmt=RAW_FMT)
            with self.assertNoLogs('name', 10):
                _ = logger.debug('msg')

    def test_info_does_not_log(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 30, fmt=RAW_FMT)
            with self.assertNoLogs('name', 20):
                _ = logger.info('msg')

    def test_warning_logs(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 30, fmt=RAW_FMT)
            _ = logger.warning('msg')
            self.assertEqual(b'msg\n', file.read())

    def test_error_logs(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 30, fmt=RAW_FMT)
            _ = logger.error('msg')
            self.assertEqual(b'msg\n', file.read())

    def test_critical_logs(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name, 30, fmt=RAW_FMT)
            _ = logger.critical('msg')
            self.assertEqual(b'msg\n', file.read())


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            excepted = (f"FileLogger('{file.name}', 'a', 10,"
                        f" '{SHORT_FMT}', 'utf-8', True)")
            self.assertEqual(excepted, repr(logger))

    def test_custom_repr(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(
                file.name,
                30,
                RAW_FMT,
                'w',
                'ascii',
                False
            )
            excepted = (f"FileLogger('{file.name}', 'w', 30,"
                        f" '{RAW_FMT}', 'ascii', False)")
            self.assertEqual(excepted, repr(logger))

    def test_pickle_works(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(file.name)
            _ = pickle.loads(pickle.dumps(logger))


if __name__ == '__main__':
    unittest.main()
