import sys
import json
import pickle
import unittest
import datetime as dt
from logging import LogRecord
from swak.jsonobject import JsonObject
from swak.misc import JsonStreamHandler


def serialize(mapping: dict) -> str:
    return json.dumps(
        mapping,
        sort_keys=True,
        default=lambda x: x.as_json if hasattr(x, 'as_json') else repr(x),
    )


class Custom:

    @property
    def as_json(self) -> str:
        return 'JSON'


class Child(JsonObject):
    c: int = 1
    d: bool = True


class Parent(JsonObject):
    a: str = 'foo'
    b: Child = Child()


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.handler = JsonStreamHandler()

    def test_has_stream(self):
        self.assertTrue(hasattr(self.handler, 'stream'))

    def test_stream(self):
        self.assertIs(self.handler.stream, sys.stdout)

    def test_has_field(self):
        self.assertTrue(hasattr(self.handler, 'fields'))

    def test_fields(self):
        self.assertSetEqual({'levelname', 'name'}, self.handler.fields)

    def test_has_extras(self):
        self.assertTrue(hasattr(self.handler, 'extras'))

    def test_extras(self):
        self.assertDictEqual({}, self.handler.extras)

    def test_has_basics(self):
        self.assertTrue(hasattr(self.handler, 'basics'))

    def test_basics(self):
        self.assertSetEqual({'levelname', 'name'}, self.handler.basics)

    def test_has_format(self):
        self.assertTrue(hasattr(self.handler, 'format'))

    def test_format(self):
        self.assertTrue(callable(self.handler.format))

    def test_has_asctime(self):
        self.assertTrue(hasattr(self.handler, 'asctime'))

    def test_asctime_callable(self):
        self.assertTrue(callable(self.handler.asctime))

    def test_call_asctime(self):
        timestamp = 1234.567
        datetime = dt.datetime.fromtimestamp(timestamp)
        expected = datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        actual = self.handler.asctime(timestamp)
        self.assertEqual(expected, actual)


class TestCustomAttributes(unittest.TestCase):

    def test_stream(self):
        handler = JsonStreamHandler('stderr')
        self.assertIs(handler.stream, sys.stderr)

    def test_stream_raises_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = JsonStreamHandler(1)

    def test_stream_raises_wrong_value(self):
        with self.assertRaises(ValueError):
            _ = JsonStreamHandler('stream')

    def test_allowed_field_normalized(self):
        handler = JsonStreamHandler('stderr', ' AsCtImE  ')
        self.assertSetEqual({'asctime'}, handler.fields)

    def test_allowed_fields_normalized(self):
        handler = JsonStreamHandler('stderr', [' AsCtImE  ', '  LeVelNo'])
        self.assertSetEqual({'asctime', 'levelno'}, handler.fields)

    def test_allowed_fields_deduplicated(self):
        handler = JsonStreamHandler('stderr', [' AsCtImE  ', 'asctime'])
        self.assertSetEqual({'asctime'}, handler.fields)

    def test_single_wrong_field_type_raises(self):
        with self.assertRaises(TypeError):
            _ = JsonStreamHandler('stderr', 1)

    def test_single_wrong_field_value_raises(self):
        with self.assertRaises(ValueError):
            _ = JsonStreamHandler('stderr', 'invalid')

    def test_one_wrong_field_type_raises(self):
        with self.assertRaises(ValueError):
            _ = JsonStreamHandler('stderr', [1, 'name'])

    def test_one_wrong_field_value_raises(self):
        with self.assertRaises(ValueError):
            _ = JsonStreamHandler('stderr', ['invalid', 'name'])

    def test_only_additional_fields_normalized(self):
        handler = JsonStreamHandler('stderr', (), ' AsCtImE  ', 'LeVelNo')
        self.assertSetEqual({'asctime', 'levelno'}, handler.fields)

    def test_only_additional_fields_deduplicted(self):
        handler = JsonStreamHandler('stderr', (), ' AsCtImE  ', 'asctime')
        self.assertSetEqual({'asctime'}, handler.fields)

    def test_only_wrong_additional_field_type_raises(self):
        with self.assertRaises(ValueError):
            _ = JsonStreamHandler('stderr', (), 1)

    def test_only_wrong_additional_field_value_raises(self):
        with self.assertRaises(ValueError):
            _ = JsonStreamHandler('stderr', (), 'invalid')

    def test_fields_additional_fields_deduplicated(self):
        handler = JsonStreamHandler(
            'stderr',
            ('asctime', 'name'),
            'name',
            'levelno'
        )
        self.assertSetEqual({'asctime', 'name', 'levelno'}, handler.fields)

    def test_basics_equals_fields(self):
        handler = JsonStreamHandler('stderr', 'asctime', 'name')
        self.assertSetEqual({'asctime', 'name'}, handler.basics)

    def test_message_stripped_from_basics(self):
        handler = JsonStreamHandler('stderr', 'asctime', 'message')
        self.assertSetEqual({'asctime', 'message'}, handler.fields)
        self.assertSetEqual({'asctime'}, handler.basics)

    def test_extras(self):
        handler = JsonStreamHandler(answer=42)
        self.assertDictEqual({'answer': 42}, handler.extras)


class TestStringFormat(unittest.TestCase):

    def setUp(self):
        self.name = 'logger'
        self.level = 20
        self.pathname = '/path/name'
        self.lineno = 42
        self.msg = 'in a bottle'
        self.func = 'func'
        self.record = LogRecord(
            name=self.name,
            level=self.level,
            pathname=self.pathname,
            lineno=self.lineno,
            msg=self.msg,
            args=(),
            exc_info=(None, None, None),
            func=self.func
        )

    def test_defaults(self):
        handler = JsonStreamHandler()
        actual = handler.format(self.record)
        expected = {
            'levelname': 'INFO',
            'message': self.msg,
            'name': self.name
        }
        self.assertEqual(serialize(expected), actual)

    def test_asctime(self):
        handler = JsonStreamHandler(field='asctime')
        actual = handler.format(self.record)
        expected = {
            'message': self.msg,
            'asctime': handler.asctime(self.record.created),
        }
        self.assertEqual(serialize(expected), actual)

    def test_explicit_message(self):
        handler = JsonStreamHandler('stdout', 'asctime', 'message')
        actual = handler.format(self.record)
        expected = {
            'message': self.msg,
            'asctime': handler.asctime(self.record.created),
        }
        self.assertEqual(serialize(expected), actual)

    def test_extras(self):
        handler = JsonStreamHandler(answer=42)
        actual = handler.format(self.record)
        expected = {
            'levelname': 'INFO',
            'message': self.msg,
            'name': self.name,
            'answer':42
        }
        self.assertEqual(serialize(expected), actual)

    def test_extras_overwritten(self):
        handler = JsonStreamHandler(answer=42, name='will be overwritten')
        actual = handler.format(self.record)
        expected = {
            'levelname': 'INFO',
            'message': self.msg,
            'name': self.name,
            'answer':42
        }
        self.assertEqual(serialize(expected), actual)


class TestDictFormat(unittest.TestCase):

    def setUp(self):
        self.name = 'logger'
        self.level = 20
        self.pathname = '/path/name'
        self.lineno = 42
        self.msg = {'answer': 42}
        self.func = 'func'
        self.record = LogRecord(
            name=self.name,
            level=self.level,
            pathname=self.pathname,
            lineno=self.lineno,
            msg=self.msg,
            args=(),
            exc_info=(None, None, None),
            func=self.func
        )

    def test_defaults(self):
        handler = JsonStreamHandler()
        actual = handler.format(self.record)
        expected = {
            'levelname': 'INFO',
            'name': self.name,
            **self.msg
        }
        self.assertEqual(serialize(expected), actual)

    def test_asctime(self):
        handler = JsonStreamHandler(field='asctime')
        actual = handler.format(self.record)
        expected = {
            'asctime': handler.asctime(self.record.created),
            **self.msg
        }
        self.assertEqual(serialize(expected), actual)

    def test_explicit_message(self):
        handler = JsonStreamHandler('stdout', 'asctime', 'message')
        actual = handler.format(self.record)
        expected = {
            'message': self.msg,
            'asctime': handler.asctime(self.record.created),
        }
        self.assertEqual(serialize(expected), actual)

    def test_message_overwritten(self):
        handler = JsonStreamHandler('stdout', 'name')
        self.record.msg = {**self.msg, 'name': 'will be overwritten'}
        actual = handler.format(self.record)
        expected = {
            'name': self.name,
            **self.msg
        }
        self.assertEqual(serialize(expected), actual)

    def test_extras(self):
        handler = JsonStreamHandler(foo='bar')
        actual = handler.format(self.record)
        expected = {
            'levelname': 'INFO',
            'name': self.name,
            **self.msg,
            'foo': 'bar'
        }
        self.assertEqual(serialize(expected), actual)

    def test_extras_overwritten(self):
        handler = JsonStreamHandler(answer='will be overwritten')
        actual = handler.format(self.record)
        expected = {
            'levelname': 'INFO',
            'name': self.name,
            **self.msg
        }
        self.assertEqual(serialize(expected), actual)

    def test_serialization_default(self):
        handler = JsonStreamHandler()
        obj = object()
        self.record.msg = {**self.msg, 'object': obj}
        actual = handler.format(self.record)
        expected = {
            'levelname': 'INFO',
            'name': self.name,
            **self.msg,
            'object': repr(obj)
        }
        self.assertEqual(serialize(expected), actual)

    def test_serialization_custom(self):
        handler = JsonStreamHandler()
        obj = Custom()
        self.record.msg = {**self.msg, 'object': obj}
        actual = handler.format(self.record)
        expected = {
            'levelname': 'INFO',
            'name': self.name,
            **self.msg,
            'object': 'JSON'
        }
        self.assertEqual(serialize(expected), actual)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.name = 'logger'
        self.level = 20
        self.pathname = '/path/name'
        self.lineno = 42
        self.msg = {'answer': 42}
        self.func = 'func'
        self.record = LogRecord(
            name=self.name,
            level=self.level,
            pathname=self.pathname,
            lineno=self.lineno,
            msg=Parent(),
            args=(),
            exc_info=(None, None, None),
            func=self.func
        )

    def test_json_object(self):
        handler = JsonStreamHandler('stdout', 'levelname')
        actual = handler.format(self.record)
        expected = {'levelname': 'INFO', **Parent()}
        self.assertEqual(serialize(expected), actual)

    def test_pickle_fails(self):
        handler = JsonStreamHandler()
        with self.assertRaises(TypeError):
            _ = pickle.loads(pickle.dumps(handler))


if __name__ == '__main__':
    unittest.main()
