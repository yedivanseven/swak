import pickle
import unittest
from unittest.mock import Mock
from swak.pl import Sort


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.by = 'col'
        self.sort = Sort(self.by)

    def test_has_by(self):
        self.assertTrue(hasattr(self.sort, 'by'))

    def test_by(self):
        self.assertEqual(self.by, self.sort.by)

    def test_has_more_by(self):
        self.assertTrue(hasattr(self.sort, 'more_by'))

    def test_more_by(self):
        self.assertTupleEqual((), self.sort.more_by)

    def test_has_descending(self):
        self.assertTrue(hasattr(self.sort, 'descending'))

    def test_descending(self):
        self.assertIs(self.sort.descending, False)

    def test_has_nulls_last(self):
        self.assertTrue(hasattr(self.sort, 'nulls_last'))

    def test_nulls_last(self):
        self.assertIs(self.sort.nulls_last, False)

    def test_has_multithreaded(self):
        self.assertTrue(hasattr(self.sort, 'multithreaded'))

    def test_multithreaded(self):
        self.assertIs(self.sort.multithreaded, True)

    def test_has_maintain_order(self):
        self.assertTrue(hasattr(self.sort, 'maintain_order'))

    def test_maintain_order(self):
        self.assertIs(self.sort.maintain_order, False)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.by = 'col1'
        self.more_by = 'col2', 'col3'
        self.descending = True
        self.nulls_last = True
        self.multithreaded = False
        self.maintain_order = True
        self.sort = Sort(
            self.by,
            *self.more_by,
            descending=self.descending,
            nulls_last=self.nulls_last,
            multithreaded=self.multithreaded,
            maintain_order=self.maintain_order,
        )

    def test_by(self):
        self.assertEqual(self.by, self.sort.by)

    def test_more_by(self):
        self.assertTupleEqual(self.more_by, self.sort.more_by)

    def test_descending(self):
        self.assertIs(self.sort.descending, self.descending)

    def test_nulls_last(self):
        self.assertIs(self.sort.nulls_last, self.nulls_last)

    def test_multithreaded(self):
        self.assertIs(self.sort.multithreaded, self.multithreaded)

    def test_maintain_order(self):
        self.assertIs(self.sort.maintain_order, self.maintain_order)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.by = 'col1'
        self.more_by = 'col2', 'col3'
        self.descending = True
        self.nulls_last = True
        self.multithreaded = False
        self.maintain_order = True
        self.sort = Sort(
            self.by,
            *self.more_by,
            descending=self.descending,
            nulls_last=self.nulls_last,
            multithreaded=self.multithreaded,
            maintain_order=self.maintain_order
        )

    def test_callable(self):
        self.assertTrue(callable(self.sort))

    def test_sort_called(self):
        df = Mock()
        _ = self.sort(df)
        df.sort.assert_called_once_with(
            self.by,
            *self.more_by,
            descending=self.descending,
            nulls_last=self.nulls_last,
            multithreaded=self.multithreaded,
            maintain_order=self.maintain_order
        )

    def test_return_value(self):
        df = Mock()
        df.sort = Mock(return_value='answer')
        actual = self.sort(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        sort = Sort('col')
        expected = ("Sort('col', descending=False, nulls_last=False, "
                    "multithreaded=True, maintain_order=False)")
        self.assertEqual(expected, repr(sort))

    def test_custom_repr(self):
        sort = Sort(
            'col1',
            'col2',
            'col3',
            descending=True,
            nulls_last=True,
            multithreaded=False,
            maintain_order=True,
        )
        expected = ("Sort('col1', 'col2', 'col3', "
                    "descending=True, nulls_last=True, "
                    "multithreaded=False, maintain_order=True)")
        self.assertEqual(expected, repr(sort))

    def test_pickle_works(self):
        sort = Sort('col')
        _ = pickle.loads(pickle.dumps(sort))


if __name__ == '__main__':
    unittest.main()
