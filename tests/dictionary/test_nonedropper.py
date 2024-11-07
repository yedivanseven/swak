import unittest
import pickle
from swak.dictionary import NoneDropper


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.drop = NoneDropper()

    def test_has_mapping(self):
        self.assertTrue(hasattr(self.drop, 'mapping'))

    def test_mapping(self):
        self.assertIsNone(self.drop.mapping)

    def test_has_recursive(self):
        self.assertTrue(hasattr(self.drop, 'recursive'))

    def test_recursive(self):
        self.assertTrue(callable(self.drop.recursive))


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mapping = {1: '1', 2: None, 3: '3', 4: None}
        self.drop = NoneDropper(self.mapping)

    def test_mapping(self):
        self.assertDictEqual(self.mapping, self.drop.mapping)


class TestDefaultUsage(unittest.TestCase):

    def setUp(self):
        self.drop = NoneDropper()

    def test_callable(self):
        self.assertTrue(callable(self.drop))

    def test_empty_raises(self):
        with self.assertRaises(TypeError):
            _ = self.drop()

    def test_none_raises(self):
        with self.assertRaises(TypeError):
            _ = self.drop(None)

    def test_non_dict_raises(self):
        with self.assertRaises(TypeError):
            _ = self.drop(1.0)

    def test_empty_dict(self):
        actual = self.drop({})
        self.assertDictEqual({}, actual)

    def test_no_nones_dict(self):
        expected = {1: '1', 3: '3'}
        actual = self.drop(expected)
        self.assertDictEqual(expected, actual)

    def test_nones_dict(self):
        expected = {1: '1', 3: '3'}
        actual = self.drop({1: '1', 2: None, 3: '3', 4: None})
        self.assertDictEqual(expected, actual)

    def test_recursive_object(self):
        expected = object()
        actual = self.drop.recursive(expected)
        self.assertIs(actual, expected)

    def test_recursive_empty_dict(self):
        actual = self.drop({})
        self.assertDictEqual({}, actual)

    def test_recursive_no_nones_dict(self):
        expected = {1: '1', 3: '3'}
        actual = self.drop.recursive(expected)
        self.assertDictEqual(expected, actual)

    def test_recursive_nones_dict(self):
        expected = {1: '1', 3: '3'}
        actual = self.drop.recursive({1: '1', 2: None, 3: '3', 4: None})
        self.assertDictEqual(expected, actual)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mapping = {1: '1', 2: None, 3: '3', 4: None}
        self.drop = NoneDropper(self.mapping)
        self.expected = {1: '1', 3: '3'}

    def test_empty(self):
        actual = self.drop()
        self.assertDictEqual(self.expected, actual)

    def test_none(self):
        actual = self.drop(None)
        self.assertDictEqual(self.expected, actual)

    def test_non_dict_raises(self):
        with self.assertRaises(TypeError):
            _ = self.drop(1.0)

    def test_empty_dict(self):
        actual = self.drop({})
        self.assertDictEqual({}, actual)

    def test_no_nones_dict(self):
        expected = {1: '1', 3: '3'}
        actual = self.drop(expected)
        self.assertDictEqual(expected, actual)

    def test_nones_dict(self):
        expected = {5: '1', 6: '3'}
        actual = self.drop({5: '1', 2: None, 6: '3', 4: None})
        self.assertDictEqual(expected, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        drop = NoneDropper()
        expected = 'NoneDropper()'
        self.assertEqual(expected, repr(drop))

    def test_custom_repr(self):
        drop = NoneDropper({'hello': 'world', 'answer': None})
        expected = 'NoneDropper()'
        self.assertEqual(expected, repr(drop))

    def test_pickle_works(self):
        drop = NoneDropper({'hello': 'world', 'answer': None})
        _ = pickle.dumps(drop)


if __name__ == '__main__':
    unittest.main()
