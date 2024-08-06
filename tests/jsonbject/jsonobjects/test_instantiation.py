import unittest
import pandas as pd
from swak.jsonobject import JsonObjects, JsonObject
from swak.jsonobject.exceptions import ParseError, ValidationErrors


class Item(JsonObject):
    a: int = 1
    b: str = 'foo'


class Items(JsonObjects, item_type=Item):
    pass


class Extra(JsonObject, ignore_extra=False, raise_extra=False):
    a: int = 1
    b: str = 'foo'


class Extras(JsonObjects, item_type=Extra):
    pass


class TestEmpty(unittest.TestCase):

    def test_empty(self):
        items = Items()
        self.assertTupleEqual((), tuple(items))

    def test_empty_list(self):
        items = Items([])
        self.assertTupleEqual((), tuple(items))

    def test_none(self):
        items = Items(None)
        self.assertTupleEqual((), tuple(items))

    def test_empty_str(self):
        items = Items('')
        self.assertTupleEqual((), tuple(items))

    def test_empty_bytes(self):
        items = Items(''.encode())
        self.assertTupleEqual((), tuple(items))

    def test_empty_bytearray(self):
        items = Items(bytearray(''.encode()))
        self.assertTupleEqual((), tuple(items))

    def test_str_empty_list(self):
        items = Items('[]')
        self.assertTupleEqual((), tuple(items))

    def test_byte_empty_list(self):
        items = Items('[]'.encode())
        self.assertTupleEqual((), tuple(items))

    def test_bytearray_empty_list(self):
        items = Items(bytearray('[]'.encode()))
        self.assertTupleEqual((), tuple(items))

    def test_empty_dataframe(self):
        items = Items(pd.DataFrame())
        self.assertTupleEqual((), tuple(items))


class TestNonEmpty(unittest.TestCase):

    def setUp(self) -> None:
        self.item1 = {'a': 1, 'b': 'foo'}
        self.item2 = {'a': 2, 'b': 'bar'}
        self.item3 = {'a': 3, 'b': 'baz'}
        self.item4 = {'a': 4, 'b': 'hello world'}

    def test_list_of_dict(self):
        _ = Items([self.item1, self.item2, self.item3, self.item4])

    def test_list_of_str(self):
        item1 = '{"a": 1, "b": "foo"}'
        item2 = '{"a": 2, "b": "bar"}'
        item3 = '{"a": 3, "b": "baz"}'
        item4 = '{"a": 4, "b": "hello world"}'
        _ = Items([item1, item2, item3, item4])

    def test_str_with_list(self):
        item1 = '{"a": 1, "b": "foo"}'
        item2 = '{"a": 2, "b": "bar"}'
        item3 = '{"a": 3, "b": "baz"}'
        item4 = '{"a": 4, "b": "hello world"}'
        items = f'[{item1}, {item2}, {item3}, {item4}]'
        _ = Items(items)

    def test_list_jsonobjects(self):
        _ = Items([Item(self.item1), Item(self.item2), Item(self.item3)])

    def test_self(self):
        items = Items([self.item1, self.item2, self.item3, self.item4])
        _ = Items(items)

    def test_dataframe(self):
        items = pd.DataFrame([self.item1, self.item2, self.item3, self.item4])
        _ = Items(items)

    def test_mix(self):
        item2 = '{"a": 2, "b": "bar"}'
        _ = Items([self.item1, item2, Item(self.item3)])


class TestEmptyArgs(unittest.TestCase):

    def setUp(self) -> None:
        self.item1 = {'a': 1, 'b': 'foo'}
        self.item2 = {'a': 2, 'b': 'bar'}
        self.item3 = {'a': 3, 'b': 'baz'}
        self.item4 = {'a': 4, 'b': 'hello world'}

    def test_empty_list_and_dict_args(self):
        _ = Items([], self.item1, self.item2, self.item3, self.item4)

    def test_none_and_dict_args(self):
        _ = Items(None, self.item1, self.item2, self.item3, self.item4)

    def test_empty_str_and_dict_args(self):
        _ = Items('', self.item1, self.item2, self.item3, self.item4)

    def test_str_of_empty_list_and_dict_args(self):
        _ = Items('[]', self.item1, self.item2, self.item3, self.item4)

    def test_empty_dataframe_and_dict_args(self):
        _ = Items(pd.DataFrame(), self.item1, self.item2, self.item3)

    def test_none_and_str_args(self):
        item1 = '{"a": 1, "b": "foo"}'
        item2 = '{"a": 2, "b": "bar"}'
        item3 = '{"a": 3, "b": "baz"}'
        item4 = '{"a": 4, "b": "hello world"}'
        _ = Items(None, item1, item2, item3, item4)

    def test_none_and_jsonobject_args(self):
        _ = Items(None, Item(self.item1), Item(self.item2), Item(self.item3))

    def test_none_and_mixed_args(self):
        item2 = '{"a": 2, "b": "bar"}'
        _ = Items(None, self.item1, item2, Item(self.item3))


class TestArgsOnly(unittest.TestCase):

    def test_str_args_only(self):
        item1 = '{"a": 1, "b": "foo"}'
        item2 = '{"a": 2, "b": "bar"}'
        _ = Items(item1, item2)

    def test_str_and_dict_args_only(self):
        item1 = '{"a": 1, "b": "foo"}'
        item2 = {"a": 2, "b": "bar"}
        _ = Items(item1, item2)

    def test_str_and_jsonobject_args_only(self):
        item1 = '{"a": 1, "b": "foo"}'
        item2 = Item({"a": 2, "b": "bar"})
        _ = Items(item1, item2)

    def test_jsonobject_args_only(self):
        item1 = Item({"a": 1, "b": "foo"})
        item2 = Item({"a": 2, "b": "bar"})
        _ = Items(item1, item2)

    def test_jsonobject_and_dict_args_only(self):
        item1 = Item({"a": 1, "b": "foo"})
        item2 = {"a": 2, "b": "bar"}
        _ = Items(item1, item2)

    def test_jsonobject_and_str_args_only(self):
        item1 = Item({"a": 1, "b": "foo"})
        item2 = '{"a": 2, "b": "bar"}'
        _ = Items(item1, item2)

    def test_dict_args_only(self):
        item1 = {"a": 1, "b": "foo"}
        item2 = {"a": 2, "b": "bar"}
        _ = Items(item1, item2)

    def test_dict_and_jsonobject_args_only(self):
        item1 = {"a": 1, "b": "foo"}
        item2 = Item({"a": 2, "b": "bar"})
        _ = Items(item1, item2)

    def test_dict_and_str_args_only(self):
        item1 = {"a": 1, "b": "foo"}
        item2 = '{"a": 2, "b": "bar"}'
        _ = Items(item1, item2)


class TestNonEmptyArgs(unittest.TestCase):

    def setUp(self) -> None:
        self.item1 = {'a': 1, 'b': 'foo'}
        self.item2 = {'a': 2, 'b': 'bar'}
        self.item3 = {'a': 3, 'b': 'baz'}
        self.item4 = {'a': 4, 'b': 'hello world'}

    def test_list_of_dict_with_dict_args(self):
        _ = Items([self.item1, self.item2], self.item3, self.item4)

    def test_list_of_dict_with_str_args(self):
        item3 = '{"a": 3, "b": "baz"}'
        item4 = '{"a": 4, "b": "hello world"}'
        _ = Items([self.item1, self.item2], item3, item4)

    def test_list_of_dict_with_jsonobject_args(self):
        _ = Items([self.item1, self.item2], Item(self.item3), Item(self.item4))

    def test_list_of_str_with_dict_args(self):
        item1 = '{"a": 1, "b": "foo"}'
        item2 = '{"a": 2, "b": "bar"}'
        _ = Items([item1, item2], self.item3, self.item4)

    def test_list_of_str_with_str_args(self):
        item1 = '{"a": 1, "b": "foo"}'
        item2 = '{"a": 2, "b": "bar"}'
        item3 = '{"a": 3, "b": "baz"}'
        item4 = '{"a": 4, "b": "hello world"}'
        _ = Items([item1, item2], item3, item4)

    def test_list_of_str_with_jsonobject_args(self):
        item1 = '{"a": 1, "b": "foo"}'
        item2 = '{"a": 2, "b": "bar"}'
        _ = Items([item1, item2], Item(self.item3), Item(self.item4))

    def test_str_with_list_with_dict_args(self):
        item1 = '{"a": 1, "b": "foo"}'
        item2 = '{"a": 2, "b": "bar"}'
        items = f'[{item1}, {item2}]'
        _ = Items(items, self.item3, self.item4)

    def test_str_with_list_with_str_args(self):
        item1 = '{"a": 1, "b": "foo"}'
        item2 = '{"a": 2, "b": "bar"}'
        item3 = '{"a": 3, "b": "baz"}'
        item4 = '{"a": 4, "b": "hello world"}'
        items = f'[{item1}, {item2}]'
        _ = Items(items, item3, item4)

    def test_str_with_list_with_jsonobject_args(self):
        item1 = '{"a": 1, "b": "foo"}'
        item2 = '{"a": 2, "b": "bar"}'
        items = f'[{item1}, {item2}]'
        _ = Items(items, Item(self.item3), Item(self.item4))

    def test_list_jsonobjects_with_dict_args(self):
        _ = Items([Item(self.item1), Item(self.item2)], self.item3)

    def test_list_jsonobjects_with_str_args(self):
        item3 = '{"a": 3, "b": "baz"}'
        item4 = '{"a": 4, "b": "hello world"}'
        _ = Items([Item(self.item1), Item(self.item2)], item3, item4)

    def test_list_jsonobjects_with_jsonobject_args(self):
        _ = Items([Item(self.item1), Item(self.item2)], Item(self.item3))

    def test_self_with_dict_args(self):
        items = Items([self.item1, self.item2])
        _ = Items(items, self.item3, self.item4)

    def test_self_with_str_args(self):
        item3 = '{"a": 3, "b": "baz"}'
        item4 = '{"a": 4, "b": "hello world"}'
        items = Items([self.item1, self.item2])
        _ = Items(items, item3, item4)

    def test_self_with_jsonobject_args(self):
        items = Items([self.item1, self.item2])
        _ = Items(items, Item(self.item3), Item(self.item4))

    def test_dataframe_with_dict_args(self):
        items = pd.DataFrame([self.item1, self.item2])
        _ = Items(items, self.item3, self.item4)

    def test_dataframe_with_str_args(self):
        item3 = '{"a": 3, "b": "baz"}'
        item4 = '{"a": 4, "b": "hello world"}'
        items = pd.DataFrame([self.item1, self.item2])
        _ = Items(items, item3, item4)

    def test_dataframe_with_jsonobject_args(self):
        items = pd.DataFrame([self.item1, self.item2])
        _ = Items(items, Item(self.item3), Item(self.item4))

    def test_list_of_dict_with_mixed_args(self):
        item3 = '{"a": 3, "b": "baz"}'
        _ = Items([self.item1], self.item2, item3, Item(self.item4))


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
