import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import Interpolate


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.interpolate = Interpolate()

    def test_has_method(self):
        self.assertTrue(hasattr(self.interpolate, 'method'))

    def test_method(self):
        self.assertEqual('linear', self.interpolate.method)

    def test_has_axis(self):
        self.assertTrue(hasattr(self.interpolate, 'axis'))

    def test_axis(self):
        self.assertEqual(0, self.interpolate.axis)

    def test_has_limit(self):
        self.assertTrue(hasattr(self.interpolate, 'limit'))

    def test_limit(self):
        self.assertIsNone(self.interpolate.limit)

    def test_has_limit_direction(self):
        self.assertTrue(hasattr(self.interpolate, 'limit_direction'))

    def test_limit_direction(self):
        self.assertIsNone(self.interpolate.limit_direction)

    def test_has_limit_area(self):
        self.assertTrue(hasattr(self.interpolate, 'limit_area'))

    def test_limit_area(self):
        self.assertIsNone(self.interpolate.limit_area)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.interpolate, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.interpolate.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.method = 'polynomial'
        self.axis = 1
        self.limit = 3
        self.limit_direction = 'both'
        self.limit_area = 'inside'
        self.kwargs = {'order': 2}
        self.interpolate = Interpolate(
            self.method,
            self.axis,
            self.limit,
            self.limit_direction,
            self.limit_area,
            **self.kwargs
        )

    def test_method(self):
        self.assertEqual(self.method, self.interpolate.method)

    def test_axis(self):
        self.assertEqual(self.axis, self.interpolate.axis)

    def test_limit(self):
        self.assertEqual(self.limit, self.interpolate.limit)

    def test_limit_direction(self):
        self.assertEqual(
            self.limit_direction,
            self.interpolate.limit_direction
        )

    def test_limit_area(self):
        self.assertEqual(self.limit_area, self.interpolate.limit_area)

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.interpolate.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.method = 'polynomial'
        self.axis = 1
        self.limit = 3
        self.limit_direction = 'both'
        self.limit_area = 'inside'
        self.kwargs = {'order': 2}
        self.interpolate = Interpolate(
            self.method,
            self.axis,
            self.limit,
            self.limit_direction,
            self.limit_area,
            **self.kwargs
        )

    def test_dataframe(self):
        df = pd.DataFrame(range(10))
        df.interpolate = Mock(return_value='expected')
        actual = self.interpolate(df)
        df.interpolate.assert_called_once_with(
            self.method,
            axis=self.axis,
            limit=self.limit,
            inplace=False,
            limit_direction=self.limit_direction,
            limit_area=self.limit_area,
            **self.kwargs
        )
        self.assertEqual('expected', actual)

    def test_series(self):
        df = pd.Series(range(10))
        df.interpolate = Mock(return_value='expected')
        actual = self.interpolate(df)
        df.interpolate.assert_called_once_with(
            self.method,
            axis=0,
            limit=self.limit,
            inplace=False,
            limit_direction=self.limit_direction,
            limit_area=self.limit_area,
            **self.kwargs
        )
        self.assertEqual('expected', actual)

    def test_resampler(self):
        df = pd.DataFrame(
            range(10),
            index=pd.bdate_range('2012-01-01', periods=10)
        ).resample('1D')
        df.interpolate = Mock(return_value='expected')
        actual = self.interpolate(df)
        df.interpolate.assert_called_once_with(
            self.method,
            axis=self.axis,
            limit=self.limit,
            inplace=False,
            limit_direction=self.limit_direction,
            limit_area=self.limit_area,
            **self.kwargs
        )
        self.assertEqual('expected', actual)

    def test_raises_on_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = self.interpolate(2)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        interpolate = Interpolate()
        expected = "Interpolate('linear', 0, None, None, None)"
        self.assertEqual(expected, repr(interpolate))

    def test_custom_repr(self):
        interpolate = Interpolate(
            'polynomial',
            1,
            3,
            'both',
            'inside',
            order=2
        )
        expected = ("Interpolate('polynomial', 1, 3, 'both', 'inside', "
                     "order=2)")
        self.assertEqual(expected, repr(interpolate))

    def test_pickle_works(self):
        interpolate = Interpolate()
        _ = pickle.loads(pickle.dumps(interpolate))


if __name__ == '__main__':
    unittest.main()
