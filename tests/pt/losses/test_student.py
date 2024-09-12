import unittest
import math
import torch as pt
from swak.pt.losses import StudentLoss, _BaseLoss


class TestStudent(unittest.TestCase):

    def setUp(self) -> None:
        self.loc = pt.tensor([[3.], [0.]])
        self.scale = pt.tensor([[0.5], [2.0]])
        self.df = pt.tensor([[4.], [1.]])
        self.y = pt.tensor([[3.], [2.]])
        self.expected = pt.tensor([[4./3.], [4. * math.pi]]).log()

    def test_is_loss(self):
        self.assertIsInstance(StudentLoss(), _BaseLoss)

    def test_default(self):
        loss = StudentLoss()
        actual = loss(self.df, self.loc, self.scale, self.y)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_mean(self):
        loss = StudentLoss('mean')
        actual = loss(self.df, self.loc, self.scale, self.y)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_sum(self):
        loss = StudentLoss('sum')
        actual = loss(self.df, self.loc, self.scale, self.y)
        pt.testing.assert_close(actual, self.expected.sum())

    def test_none(self):
        loss = StudentLoss('none')
        actual = loss(self.df, self.loc, self.scale, self.y)
        pt.testing.assert_close(actual, self.expected)

    def test_0_loc(self):
        loss = StudentLoss('none')
        loc = pt.tensor([[0.0], [0.0]])
        actual = loss(self.df, loc, self.scale, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_scale(self):
        loss = StudentLoss('none')
        scale = pt.tensor([[0.0], [0.0]])
        actual = loss(self.df, self.loc, scale, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_df(self):
        loss = StudentLoss('none')
        df = pt.tensor([[0.0], [0.0]])
        actual = loss(df, self.loc, self.scale, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_target(self):
        loss = StudentLoss('none')
        y = pt.tensor([[0.0], [0.0]])
        actual = loss(self.df, self.loc, self.scale, y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_loc_0_scale(self):
        loss = StudentLoss('none')
        loc = pt.tensor([[0.0], [0.0]])
        scale = pt.tensor([[0.0], [0.0]])
        actual = loss(self.df, loc, scale, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_loc_0_df(self):
        loss = StudentLoss('none')
        loc = pt.tensor([[0.0], [0.0]])
        df = pt.tensor([[0.0], [0.0]])
        actual = loss(df, loc, self.scale, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_loc_0_target(self):
        loss = StudentLoss('none')
        loc = pt.tensor([[0.0], [0.0]])
        actual = loss(self.df, loc, self.scale, loc)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_scale_0_df(self):
        loss = StudentLoss('none')
        scale = pt.tensor([[0.0], [0.0]])
        df = pt.tensor([[0.0], [0.0]])
        actual = loss(df, self.loc, scale, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_scale_0_target(self):
        loss = StudentLoss('none')
        scale = pt.tensor([[0.0], [0.0]])
        actual = loss(self.df, self.loc, scale, scale)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_df_0_target(self):
        loss = StudentLoss('none')
        df = pt.tensor([[0.0], [0.0]])
        actual = loss(df, self.loc, self.scale, df)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_loc_0_scale_0_df(self):
        loss = StudentLoss('none')
        loc = pt.tensor([[0.0], [0.0]])
        scale = pt.tensor([[0.0], [0.0]])
        df = pt.tensor([[0.0], [0.0]])
        actual = loss(df, loc, scale, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_loc_0_scale_0_target(self):
        loss = StudentLoss('none')
        loc = pt.tensor([[0.0], [0.0]])
        scale= pt.tensor([[0.0], [0.0]])
        actual = loss(self.df, loc, scale, loc)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_loc_0_df_0_target(self):
        loss = StudentLoss('none')
        loc = pt.tensor([[0.0], [0.0]])
        df = pt.tensor([[0.0], [0.0]])
        actual = loss(df, loc, self.scale, loc)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_scale_0_df_0_target(self):
        loss = StudentLoss('none')
        scale = pt.tensor([[0.0], [0.0]])
        df = pt.tensor([[0.0], [0.0]])
        actual = loss(df, self.loc, scale, scale)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_loc_0_scale_0_df_0_target(self):
        loss = StudentLoss('none')
        loc = pt.tensor([[0.0], [0.0]])
        scale = pt.tensor([[0.0], [0.0]])
        df = pt.tensor([[0.0], [0.0]])
        actual = loss(df, loc, scale, loc)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())


if __name__ == '__main__':
    unittest.main()
