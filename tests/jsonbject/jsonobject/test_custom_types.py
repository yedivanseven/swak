import unittest
from unittest.mock import Mock
import pandas as pd
from swak.jsonobject import JsonObject


class Dummy:

    def __init__(self, *_):
        pass


class DefinesAsJson:

    def __init__(self, *_):
        pass

    @property
    def as_json(self):
        return 'mocked as_json'


class DefinesAsDtype:

    def __init__(self, *_):
        pass

    @property
    def as_dtype(self):
        return 'mocked as_dtype'


class TestCustomTypeUndefined(unittest.TestCase):

    def test_repr_calls_repr(self):
        mock = Mock(return_value='mocked repr')
        Dummy.__repr__ = mock

        class Custom(JsonObject):
            d: Dummy

        custom = Custom(d=1)
        r = repr(custom)
        mock.assert_called_once()
        self.assertEqual('{\n    "d": "mocked repr"\n}', r)

    def test_str_calls_repr(self):
        mock = Mock(return_value='mocked repr')
        Dummy.__repr__ = mock

        class Custom(JsonObject):
            d: Dummy

        custom = Custom(d=1)
        s = str(custom)
        mock.assert_called_once()
        self.assertEqual('{"d": "mocked repr"}', s)

    def test_repr_calls_as_json(self):

        class Custom(JsonObject):
            d: DefinesAsJson

        custom = Custom(d=1)
        r = repr(custom)
        self.assertEqual('{\n    "d": "mocked as_json"\n}', r)

    def test_str_calls_as_json(self):

        class Custom(JsonObject):
            d: DefinesAsJson

        custom = Custom(d=1)
        s = str(custom)
        self.assertEqual('{"d": "mocked as_json"}', s)

    def test_as_json_calls_repr(self):
        mock = Mock(return_value='mocked repr')
        Dummy.__repr__ = mock

        class Custom(JsonObject):
            d: Dummy

        custom = Custom(d=1)
        as_json = custom.as_json
        mock.assert_called_once()
        self.assertDictEqual({'d': 'mocked repr'}, as_json)

    def test_as_json_calls_as_json(self):

        class Custom(JsonObject):
            d: DefinesAsJson

        custom = Custom(d=1)
        as_json = custom.as_json
        self.assertDictEqual({'d': 'mocked as_json'}, as_json)

    def test_as_dtype_calls_repr(self):
        mock = Mock(return_value='mocked repr')
        Dummy.__repr__ = mock

        class Custom(JsonObject):
            d: Dummy

        custom = Custom(d=1)
        as_dtype = custom.as_dtype
        mock.assert_called_once()
        self.assertEqual('{"d": "mocked repr"}', as_dtype)

    def test_as_dtype_calls_as_json(self):

        class Custom(JsonObject):
            d: DefinesAsJson

        custom = Custom(d=1)
        as_dtype = custom.as_dtype
        self.assertEqual('{"d": "mocked as_json"}', as_dtype)

    def test_as_series_calls_as_dtype(self):

        class Custom(JsonObject):
            d: DefinesAsDtype

        custom = Custom(d=1)
        as_series = custom.as_series
        self.assertIsInstance(as_series, pd.Series)
        expected = pd.Series({'d': 'mocked as_dtype'}, name='Custom')
        pd.testing.assert_series_equal(expected, as_series)


if __name__ == '__main__':
    unittest.main()
