import pickle
import unittest
from unittest.mock import Mock
import polars as pl
from swak.pl import Join


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.join = Join()

    def test_has_on(self):
        self.assertTrue(hasattr(self.join, 'on'))

    def test_on(self):
        self.assertIsNone(self.join.on)

    def test_has_how(self):
        self.assertTrue(hasattr(self.join, 'how'))

    def test_how(self):
        self.assertEqual('inner', self.join.how)

    def test_has_left_on(self):
        self.assertTrue(hasattr(self.join, 'left_on'))

    def test_left_on(self):
        self.assertIsNone(self.join.left_on)

    def test_has_right_on(self):
        self.assertTrue(hasattr(self.join, 'right_on'))

    def test_right_on(self):
        self.assertIsNone(self.join.right_on)

    def test_has_suffix(self):
        self.assertTrue(hasattr(self.join, 'suffix'))

    def test_suffix(self):
        self.assertEqual('_right', self.join.suffix)

    def test_has_validate(self):
        self.assertTrue(hasattr(self.join, 'validate'))

    def test_validate(self):
        self.assertEqual('m:m', self.join.validate)

    def test_has_nulls_equal(self):
        self.assertTrue(hasattr(self.join, 'nulls_equal'))

    def test_nulls_equal(self):
        self.assertIs(self.join.nulls_equal, False)

    def test_has_coalesce(self):
        self.assertTrue(hasattr(self.join, 'coalesce'))

    def test_coalesce(self):
        self.assertIsNone(self.join.coalesce)

    def test_has_maintain_order(self):
        self.assertTrue(hasattr(self.join, 'maintain_order'))

    def test_maintain_order(self):
        self.assertIsNone(self.join.maintain_order)


class TestCustomAttributes(unittest.TestCase):

    def setUp(self):
        self.on = 'id'
        self.how = '  LEFT '
        self.left_on = ['id', 'name']
        self.right_on = 'id'
        self.suffix = ' _suffix '
        self.validate = ' 1:1 '
        self.nulls_equal = True
        self.coalesce = True
        self.maintain_order = 'left'
        self.join = Join(
            on=self.on,
            how=self.how,
            left_on=self.left_on,
            right_on=self.right_on,
            suffix=self.suffix,
            validate=self.validate,
            nulls_equal=self.nulls_equal,
            coalesce=self.coalesce,
            maintain_order=self.maintain_order,
        )

    def test_on(self):
        self.assertEqual(self.on, self.join.on)

    def test_how(self):
        self.assertEqual('left', self.join.how)

    def test_left_on(self):
        self.assertEqual(self.left_on, self.join.left_on)

    def test_right_on(self):
        self.assertEqual(self.right_on, self.join.right_on)

    def test_suffix(self):
        self.assertEqual('_suffix', self.join.suffix)

    def test_validate(self):
        self.assertEqual('1:1', self.join.validate)

    def test_nulls_equal(self):
        self.assertIs(self.join.nulls_equal, True)

    def test_coalesce(self):
        self.assertIs(self.join.coalesce, True)

    def test_maintain_order(self):
        self.assertEqual(self.maintain_order, self.join.maintain_order)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.on = 'id'
        self.how = 'left'
        self.left_on = 'left_id'
        self.right_on = 'right_id'
        self.suffix = '_suffix'
        self.validate = '1:m'
        self.nulls_equal = True
        self.coalesce = True
        self.maintain_order = 'left'
        self.join = Join(
            on=self.on,
            how=self.how,
            suffix=self.suffix,
            validate=self.validate,
            nulls_equal=self.nulls_equal,
            coalesce=self.coalesce,
            maintain_order=self.maintain_order,
        )

    def test_callable(self):
        self.assertTrue(callable(self.join))

    def test_join_called(self):
        left = Mock()
        right = object()
        _ = self.join(left, right)
        left.join.assert_called_once_with(
            right,
            self.on,
            self.how,
            left_on=None,
            right_on=None,
            suffix=self.suffix,
            validate=self.validate,
            nulls_equal=self.nulls_equal,
            coalesce=self.coalesce,
            maintain_order=self.maintain_order,
        )

    def test_return_value(self):
        left = Mock()
        left.join = Mock(return_value='join_result')
        right = object()
        actual = self.join(left, right)
        self.assertEqual('join_result', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        join = Join()
        expected = ("Join(None, 'inner', left_on=None, right_on=None, "
                    "suffix='_right', validate='m:m', nulls_equal=False, "
                    "coalesce=None, maintain_order=None)")
        self.assertEqual(expected, repr(join))

    def test_custom_repr(self):
        join = Join(
            on='id',
            how='left',
            suffix='_suffix',
            validate='1:1',
            nulls_equal=True,
            coalesce=True,
            maintain_order='left',
        )
        expected = ("Join('id', 'left', left_on=None, right_on=None, "
                    "suffix='_suffix', validate='1:1', nulls_equal=True, "
                    "coalesce=True, maintain_order='left')")
        self.assertEqual(expected, repr(join))

    def test_pickle_works(self):
        join = Join(
            on=pl.col('id'),
            how='left',
            suffix='_suffix',
            validate='1:1',
            nulls_equal=True,
        )
        _ = pickle.loads(pickle.dumps(join))


if __name__ == '__main__':
    unittest.main()
