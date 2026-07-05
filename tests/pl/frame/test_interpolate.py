import unittest
from unittest.mock import Mock
from swak.pl import interpolate


class TestUsage(unittest.TestCase):#

    def test_interpolate_called(self):
        df = Mock()
        df.interpolate = Mock(return_value='expected')
        actual = interpolate(df)
        df.interpolate.assert_called_once_with()
        self.assertEqual('expected', actual)



if __name__ == '__main__':
    unittest.main()
