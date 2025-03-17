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

# *more_by: IntoExpr,
# descending: bool | Sequence[bool] = False,
# nulls_last: bool | Sequence[bool] = False,
# multithreaded: bool = True,
# maintain_order: bool = False,


if __name__ == '__main__':
    unittest.main()
