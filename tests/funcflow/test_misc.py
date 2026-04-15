import unittest
from unittest.mock import Mock
from swak.funcflow import unit, identity, apply, to_list


class TestUnit(unittest.TestCase):

    def test_empty(self):
        actual = unit()
        self.assertIsInstance(actual, tuple)
        self.assertTupleEqual((), actual)

    def test_one(self):
        actual = unit('foo')
        self.assertIsInstance(actual, tuple)
        self.assertTupleEqual((), actual)

    def test_many(self):
        actual = unit(1, 'foo', False)
        self.assertIsInstance(actual, tuple)
        self.assertTupleEqual((), actual)

    def test_empty_tuple(self):
        actual = unit(())
        self.assertTupleEqual((), actual)

    def test_one_tuple(self):
        actual = unit((42,))
        self.assertTupleEqual((), actual)

    def test_two_tuple(self):
        actual = unit((42, 'foo'))
        self.assertTupleEqual((), actual)


class TestIdentity(unittest.TestCase):

    def test_empty(self):
        actual = identity()
        self.assertIsInstance(actual, tuple)
        self.assertTupleEqual((), actual)

    def test_one_arg(self):
        expected = 'foo'
        actual = identity(expected)
        self.assertIs(expected, actual)

    def test_many(self):
        expected = 1, 'foo', False
        actual = identity(*expected)
        self.assertTupleEqual(expected, actual)

    def test_empty_tuple(self):
        expected = ()
        actual = identity(expected)
        self.assertTupleEqual(expected, actual)

    def test_one_tuple(self):
        expected = 42,
        actual = identity(expected)
        self.assertTupleEqual(expected, actual)

    def test_two_tuple(self):
        expected = 42, 'foo'
        actual = identity(expected)
        self.assertTupleEqual(expected, actual)


class TestApply(unittest.TestCase):

    def test_call_called_with_no_args(self):
        mock = Mock()
        _ = apply(mock)
        mock.assert_called_once()
        mock.assert_called_once_with()

    def test_call_called_with_one_arg(self):
        mock = Mock()
        _ = apply(mock, 1)
        mock.assert_called_once()
        mock.assert_called_once_with(1)

    def test_call_called_with_multiple_args(self):
        mock = Mock()
        _ = apply(mock, 1, 'foo', False)
        mock.assert_called_once()
        mock.assert_called_once_with(1, 'foo', False)

    def test_call_called_with_empty_tuple(self):
        mock = Mock()
        _ = apply(mock, ())
        mock.assert_called_once()
        mock.assert_called_once_with(())

    def test_call_called_with_one_tuple(self):
        mock = Mock()
        _ = apply(mock, (42,))
        mock.assert_called_once()
        mock.assert_called_once_with((42,))

    def test_call_called_with_two_tuple(self):
        mock = Mock()
        _ = apply(mock, (42, 'foo'))
        mock.assert_called_once()
        mock.assert_called_once_with((42, 'foo'))


class TestToList(unittest.TestCase):

    def test_returns_list(self):
        result = to_list((1, 2, 3))
        self.assertIsInstance(result, list)

    def test_preserves_elements(self):
        result = to_list((1, 2, 3))
        self.assertEqual(result, [1, 2, 3])

    def test_preserves_order(self):
        result = to_list((3, 1, 2))
        self.assertEqual(result, [3, 1, 2])

    def test_empty_iterable(self):
        result = to_list([])
        self.assertEqual(result, [])

    def test_from_generator(self):
        result = to_list(x for x in range(3))
        self.assertEqual(result, [0, 1, 2])

    def test_from_string(self):
        result = to_list("abc")
        self.assertEqual(result, ["a", "b", "c"])

    def test_copies_list_input(self):
        original = [1, 2, 3]
        result = to_list(original)
        self.assertIsNot(result, original)


if __name__ == '__main__':
    unittest.main()
