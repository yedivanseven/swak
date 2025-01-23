import pickle
import unittest
from tempfile import NamedTemporaryFile
from logging import Logger, getLogger, FileHandler
from swak.misc import FileLogger, DEFAULT_FMT, PID_FMT, RAW_FMT


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        with NamedTemporaryFile() as file:
            _ = FileLogger('default', file.name)

    def test_has_name(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'name'))

    def test_name(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertEqual(logger.name, 'name')

    def test_name_stripped(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('  name ', file.name)
            self.assertEqual(logger.name, 'name')

    def test_has_file(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'file'))

    def test_file(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertEqual(file.name, logger.file)

    def test_has_level(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'level'))

    def test_level_type(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertIsInstance(logger.level, int)

    def test_level_value(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertEqual(10, logger.level)

    def test_has_fmt(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'fmt'))

    def test_fmt_type(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertIsInstance(logger.fmt, str)

    def test_fmt_value(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertEqual(DEFAULT_FMT, logger.fmt)

    def test_has_mode(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'mode'))

    def test_mode(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertEqual('a', logger.mode)

    def test_has_encoding(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'encoding'))

    def test_encoding(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertEqual('utf-8', logger.encoding)

    def test_has_delay(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'delay'))

    def test_delay_type(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertIsInstance(logger.delay, bool)

    def test_delay_value(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(logger.delay)

    def test_has_handler_exists(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'handler_exists'))

    def test_handler_exists_type(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertIsInstance(logger.handler_exists, bool)

    def test_handler_exists_value(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertFalse(logger.handler_exists)

    def test_has_logger(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'logger'))

    def test_logger_type(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertIsInstance(logger.logger, Logger)


class TestCustomAttributes(unittest.TestCase):

    def test_has_level(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('default', file.name, 20)
            self.assertTrue(hasattr(logger, 'level'))

    def test_level(self):
        with NamedTemporaryFile() as file:
            obj = object()
            logger = FileLogger('default', file.name, obj)
            self.assertIs(logger.level, obj)

    def test_has_fmt(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('default', file.name, 20, 'format')
            self.assertTrue(hasattr(logger, 'fmt'))

    def test_fmt(self):
        with NamedTemporaryFile() as file:
            obj = object()
            logger = FileLogger('default', file.name, 20, obj)
            self.assertIs(logger.fmt, obj)

    def test_has_mode(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, 20, PID_FMT, 'w')
            self.assertTrue(hasattr(logger, 'mode'))

    def test_mode(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, 20, PID_FMT, 'w')
            self.assertEqual('w', logger.mode)

    def test_mode_stripped(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, 20, PID_FMT, '  w ')
            self.assertEqual('w', logger.mode)

    def test_has_encoding(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, 20, PID_FMT, 'w', 'ascii')
            self.assertTrue(hasattr(logger, 'encoding'))

    def test_encoding(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, 20, PID_FMT, 'w', 'ascii')
            self.assertEqual('ascii', logger.encoding)

    def test_encoding_stripped(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, 20, PID_FMT, 'w', ' ascii')
            self.assertEqual('ascii', logger.encoding)

    def test_has_delay(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, delay=False)
            self.assertTrue(hasattr(logger, 'delay'))

    def test_delay(self):
        with NamedTemporaryFile() as file:
            obj = object
            logger = FileLogger('name', file.name, delay=obj)
            self.assertIs(logger.delay, obj)


class TestMethods(unittest.TestCase):

    def test_has_log(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'log'))

    def test_callable_log(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(callable(logger.log))

    def test_has_debug(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'debug'))

    def test_callable_debug(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(callable(logger.debug))

    def test_has_info(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'info'))

    def test_callable_info(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(callable(logger.info))

    def test_has_warning(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'warning'))

    def test_callable_warning(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(callable(logger.warning))

    def test_has_error(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'error'))

    def test_callable_error(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(callable(logger.error))

    def test_has_critical(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(hasattr(logger, 'critical'))

    def test_callable_critical(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            self.assertTrue(callable(logger.critical))


class TestUsage(unittest.TestCase):

    def test_log(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, fmt=RAW_FMT)
            actual = logger.log(10, 'msg')
            self.assertTupleEqual((), actual)
            self.assertEqual(b'msg\n', file.read())

    def test_debug(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, fmt=RAW_FMT)
            actual = logger.debug('msg')
            self.assertTupleEqual((), actual)
            self.assertEqual(b'msg\n', file.read())

    def test_info(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, fmt=RAW_FMT)
            actual = logger.info('msg')
            self.assertTupleEqual((), actual)
            self.assertEqual(b'msg\n', file.read())

    def test_warning(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, fmt=RAW_FMT)
            actual = logger.warning('msg')
            self.assertTupleEqual((), actual)
            self.assertEqual(b'msg\n', file.read())

    def test_error(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, fmt=RAW_FMT)
            actual = logger.error('msg')
            self.assertTupleEqual((), actual)
            self.assertEqual(b'msg\n', file.read())

    def test_critical(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, fmt=RAW_FMT)
            actual = logger.critical('msg')
            self.assertTupleEqual((), actual)
            self.assertEqual(b'msg\n', file.read())

    def test_handler_exists(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, fmt=RAW_FMT)
            _ = logger.critical('msg')
            self.assertTrue(logger.handler_exists)

    def test_ancestor_logger_existing_file_raises(self):
        with NamedTemporaryFile() as file:
            root = getLogger()
            root.addHandler(FileHandler(file.name))
            logger = FileLogger('name', file.name)
            with self.assertRaises(FileExistsError):
                _ = logger.info('msg')

    def test_sibling_logger_existing_file_raises(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            sibling = getLogger('sibling')
            sibling.addHandler(FileHandler(file.name))
            with self.assertRaises(FileExistsError):
                _ = logger.info('msg')

    def test_relative_logger_existing_file_raises(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            _ = getLogger('sibling')
            niece = getLogger('sibling.daughter')
            niece.addHandler(FileHandler(file.name))
            with self.assertRaises(FileExistsError):
                _ = logger.info('msg')

    def test_non_root_child_existing_file_raises(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            grandniece = getLogger('sibling.daughter.granddaughter')
            grandniece.addHandler(FileHandler(file.name))
            with self.assertRaises(FileExistsError):
                _ = logger.info('msg')

    def test_child_logger_existing_file_raises(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name)
            child = getLogger('name.child')
            child.addHandler(FileHandler(file.name))
            with self.assertRaises(FileExistsError):
                _ = logger.info('msg')


class TestLogLevel(unittest.TestCase):

    def test_debug_does_not_log(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, 30, fmt=RAW_FMT)
            with self.assertNoLogs('name', 10):
                _ = logger.debug('msg')

    def test_info_does_not_log(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, 30, fmt=RAW_FMT)
            with self.assertNoLogs('name', 20):
                _ = logger.info('msg')

    def test_warning_logs(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, 30, fmt=RAW_FMT)
            _ = logger.warning('msg')
            self.assertEqual(b'msg\n', file.read())

    def test_error_logs(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, 30, fmt=RAW_FMT)
            _ = logger.error('msg')
            self.assertEqual(b'msg\n', file.read())

    def test_critical_logs(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('name', file.name, 30, fmt=RAW_FMT)
            _ = logger.critical('msg')
            self.assertEqual(b'msg\n', file.read())


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('default', file.name)
            excepted = (f"FileLogger('default', '{file.name}', "
                        f"'a', 10, '{DEFAULT_FMT}', 'utf-8', True)")
            self.assertEqual(excepted, repr(logger))

    def test_custom_repr(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger(
                'default',
                file.name,
                30,
                RAW_FMT,
                'w',
                'ascii',
                False
            )
            excepted = (f"FileLogger('default', '{file.name}', "
                        f"'w', 30, '{RAW_FMT}', 'ascii', False)")
            self.assertEqual(excepted, repr(logger))

    def test_pickle_works(self):
        with NamedTemporaryFile() as file:
            logger = FileLogger('default', file.name)
            _ = pickle.loads(pickle.dumps(logger))


if __name__ == '__main__':
    unittest.main()
