import unittest
import pandas as pd
from swak.jsonobject import JsonObjects, JsonObject
from swak.jsonobject.exceptions import ParseError, ValidationErrors


class Item(JsonObject):
    a: int
    b: str


class Items(JsonObjects, item_type=Item):
    pass


class Extra(JsonObject, ignore_extra=False, raise_extra=False):
    a: int = 1
    b: str = 'foo'


class Extras(JsonObjects, item_type=Extra):
    pass


class Respect(
        JsonObject,
        ignore_extra=False,
        raise_extra=False,
        respect_none=True
):
    pass


class Respects(JsonObjects, item_type=Respect):
    pass


dicts = [
    {'a': 1, 'b': 'foo'},
    {'a': 2, 'b': 'bar'},
    {'a': 3, 'b': 'baz'},
    {'a': 4, 'b': 'pan'}
]

strs = [
    "{'a': 1, 'b': 'foo'}",
    "{'a': 2, 'b': 'bar'}",
    "{'a': 3, 'b': 'baz'}",
    "{'a': 4, 'b': 'pan'}"
]

jsons = [
    '{"a": 1, "b": "foo"}',
    '{"a": 2, "b": "bar"}',
    '{"a": 3, "b": "baz"}',
    '{"a": 4, "b": "pan"}'
]


class TestEmpty(unittest.TestCase):

    def test_empty(self):
        items = Items()
        self.assertListEqual([], items.as_json)

    def test_empty_list(self):
        items = Items([])
        self.assertListEqual([], items.as_json)

    def test_none(self):
        items = Items(None)
        self.assertListEqual([], items.as_json)

    def test_empty_str(self):
        items = Items('')
        self.assertListEqual([], items.as_json)

    def test_empty_bytes(self):
        items = Items(b'')
        self.assertListEqual([], items.as_json)

    def test_empty_bytearray(self):
        items = Items(bytearray(b''))
        self.assertListEqual([], items.as_json)

    def test_str_empty_list(self):
        items = Items('[]')
        self.assertListEqual([], items.as_json)

    def test_byte_empty_list(self):
        items = Items(b'[]')
        self.assertListEqual([], items.as_json)

    def test_bytearray_empty_list(self):
        items = Items(bytearray(b'[]'))
        self.assertListEqual([], items.as_json)

    def test_empty_dataframe(self):
        items = Items(pd.DataFrame())
        self.assertListEqual([], items.as_json)

    def test_empty_self(self):
        items = Items(Items())
        self.assertListEqual([], items.as_json)


class TestLists(unittest.TestCase):

    def test_list_of_dict(self):
        items = Items(dicts)
        self.assertListEqual(dicts, items.as_json)

    def test_list_of_str(self):
        items = Items(strs)
        self.assertListEqual(dicts, items.as_json)

    def test_list_of_json(self):
        items = Items(jsons)
        self.assertListEqual(dicts, items.as_json)

    def test_list_of_bytes(self):
        items = Items([j.encode() for j in jsons])
        self.assertListEqual(dicts, items.as_json)

    def test_list_of_bytearrays(self):
        items = Items([bytearray(j.encode()) for j in jsons])
        self.assertListEqual(dicts, items.as_json)

    def test_str_of_list(self):
        items = Items(f"[{strs[0]}, {strs[1]}, {strs[2]}, {strs[3]}]")
        self.assertListEqual(dicts, items.as_json)

    def test_json_of_list(self):
        items = Items(f'[{jsons[0]}, {jsons[1]}, {jsons[2]}, {jsons[3]}]')
        self.assertListEqual(dicts, items.as_json)

    def test_bytes_of_list(self):
        b = f'[{jsons[0]}, {jsons[1]}, {jsons[2]}, {jsons[3]}]'.encode()
        items = Items(b)
        self.assertListEqual(dicts, items.as_json)

    def test_bytearray_of_list(self):
        b = f'[{jsons[0]}, {jsons[1]}, {jsons[2]}, {jsons[3]}]'.encode()
        items = Items(bytearray(b))
        self.assertListEqual(dicts, items.as_json)

    def test_list_jsonobjects(self):
        items = Items([Item(d) for d in dicts])
        self.assertListEqual(dicts, items.as_json)

    def test_list_of_series(self):
        items = Items([pd.Series(d) for d in dicts])
        self.assertListEqual(dicts, items.as_json)

    def test_self(self):
        items = Items(dicts)
        items = Items(items)
        self.assertListEqual(dicts, items.as_json)

    def test_dataframe(self):
        items = Items(pd.DataFrame(dicts))
        self.assertListEqual(dicts, items.as_json)

    def test_mix(self):
        items = Items([dicts[0], strs[1], jsons[2], Item(dicts[3])])
        self.assertListEqual(dicts, items.as_json)


class TestEmptyArgs(unittest.TestCase):

    def test_empty_list_and_dict_args(self):
        items = Items([], *dicts)
        self.assertListEqual(dicts, items.as_json)

    def test_none_and_str_args(self):
        items = Items(None, *strs)
        self.assertListEqual(dicts, items.as_json)

    def test_empty_str_and_json_args(self):
        items = Items('', *jsons)
        self.assertListEqual(dicts, items.as_json)

    def test_str_of_empty_list_and_jsonobject_args(self):
        items = Items('[]', *[Item(d) for d in dicts])
        self.assertListEqual(dicts, items.as_json)

    def test_empty_dataframe_and_bytes_args(self):
        items = Items(pd.DataFrame(), *[j.encode() for j in jsons])
        self.assertListEqual(dicts, items.as_json)

    def test_empty_self_and_bytearray_args(self):
        items = Items(Items(), *[bytearray(j.encode()) for j in jsons])
        self.assertListEqual(dicts, items.as_json)

    def test_empty_bytes_and_series_args(self):
        items = Items(b'', *[bytearray(j.encode()) for j in jsons])
        self.assertListEqual(dicts, items.as_json)

    def test_empty_bytearray_and_mixed_args(self):
        items = Items(
            bytearray(b''),
            dicts[0],
            strs[1],
            jsons[2],
            Item(dicts[3])
        )
        self.assertListEqual(dicts, items.as_json)


class TestArgsOnly(unittest.TestCase):

    def test_dict_args(self):
        items = Items(*dicts)
        self.assertListEqual(dicts, items.as_json)

    def test_str_args(self):
        items = Items(*strs)
        self.assertListEqual(dicts, items.as_json)

    def test_json_args(self):
        items = Items(*jsons)
        self.assertListEqual(dicts, items.as_json)

    def test_byte_args(self):
        items = Items(*[j.encode() for j in jsons])
        self.assertListEqual(dicts, items.as_json)

    def test_bytearray_args(self):
        items = Items(*[bytearray(j.encode()) for j in jsons])
        self.assertListEqual(dicts, items.as_json)

    def test_jsonobject_args(self):
        items = Items(*[Item(d) for d in dicts])
        self.assertListEqual(dicts, items.as_json)

    def test_series_args(self):
        items = Items(*[pd.Series(d) for d in dicts])
        self.assertListEqual(dicts, items.as_json)

    def test_mixed_args(self):
        items = Items(dicts[0], strs[1], jsons[2], Item(dicts[3]))
        self.assertListEqual(dicts, items.as_json)


class TestNonEmptyArgs(unittest.TestCase):

    def test_dicts(self):
        items = Items(dicts[:2], dicts[2], dicts[3])
        self.assertListEqual(dicts, items.as_json)

    def test_strs(self):
        items = Items(strs[:2], strs[2], strs[3])
        self.assertListEqual(dicts, items.as_json)

    def test_str_and_strs(self):
        items = Items(f"[{strs[0]}, {strs[1]}]", strs[2], strs[3])
        self.assertListEqual(dicts, items.as_json)

    def test_jsons(self):
        items = Items(jsons[:2], jsons[2], jsons[3])
        self.assertListEqual(dicts, items.as_json)

    def test_json_and_jsons(self):
        items = Items(f'[{jsons[0]}, {jsons[1]}]', jsons[2], jsons[3])
        self.assertListEqual(dicts, items.as_json)

    def test_jsonobjects(self):
        items = Items(
            [Item(dicts[0]), Item(dicts[1])],
            Item(dicts[2]),
            Item(dicts[3])
        )
        self.assertListEqual(dicts, items.as_json)

    def test_self(self):
        items = Items(Items(dicts[:2]), Item(dicts[2]), Item(dicts[3]))
        self.assertListEqual(dicts, items.as_json)

    def test_dataframe(self):
        items = Items(
            pd.DataFrame(dicts[:2]),
            pd.Series(dicts[2]),
            pd.Series(dicts[3])
        )
        self.assertListEqual(dicts, items.as_json)

    def test_mixed(self):
        items = Items(
            [dicts[0], jsons[1]],
            pd.Series(dicts[2]),
            Item(dicts[3])
        )
        self.assertListEqual(dicts, items.as_json)


class TestMisc(unittest.TestCase):

    def test_extra(self):
        expected = [
            {'a': 1, 'b': 'foo'},
            {'a': 1, 'b': 'foo', 'c': 'bar'},
            {'a': 1, 'b': 'foo', 'd': 'baz'}
        ]
        extras = Extras(expected)
        self.assertListEqual(expected, extras.as_json)

    def test_respect(self):
        expected = [
            {'a': 1, 'b': 'foo'},
            {'a': 1, 'b': 'foo', 'c': 'bar', 'd': None},
            {'a': 1, 'b': 'foo', 'c': None, 'd': 'baz'}
        ]
        respects = Respects(expected)
        self.assertListEqual(expected, respects.as_json)


class TestExceptions(unittest.TestCase):

    def test_raises_raises_not_iterable(self):
        with self.assertRaises(ParseError):
            _ = Items(123)

    def test_raises_raises_not_json_string(self):
        with self.assertRaises(ParseError):
            _ = Items('123')

    def test_raises_raises_not_json_iterable(self):
        with self.assertRaises(ValidationErrors):
            _ = Items([123])

    def test_raises_raises_not_str_json_iterable(self):
        with self.assertRaises(ValidationErrors):
            _ = Items('[123]')


if __name__ == '__main__':
    unittest.main()
