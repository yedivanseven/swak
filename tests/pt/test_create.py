import unittest
from unittest.mock import patch
import torch as pt
import pandas as pd
import numpy as np
from swak.pt.create import Create, AsTensor, from_dataframe


class TestCreate(unittest.TestCase):

    def setUp(self):
        self.create = Create()

    def test_default_dtype(self):
        self.assertTrue(hasattr(self.create, 'dtype'))
        self.assertIsNone(self.create.dtype)

    def test_default_device(self):
        self.assertTrue(hasattr(self.create, 'device'))
        self.assertIsNone(self.create.device)

    def test_default_requires_grad(self):
        self.assertTrue(hasattr(self.create, 'requires_grad'))
        self.assertFalse(self.create.requires_grad)

    def test_default_pin_memory(self):
        self.assertTrue(hasattr(self.create, 'pin_memory'))
        self.assertFalse(self.create.pin_memory)

    def test_custom_args(self):
        device = pt.device('cuda')
        create = Create(pt.uint8, device, True, True)
        self.assertIs(create.dtype, pt.uint8)
        self.assertIs(create.device, device)
        self.assertTrue(create.requires_grad)
        self.assertTrue(create.pin_memory)

    def test_callable(self):
        self.assertTrue(callable(self.create))

    @patch('torch.tensor')
    def test_tensor_called(self, mock):
        _ = self.create([1, 2, 3])
        mock.assert_called_once()

    @patch('torch.tensor')
    def test_tensor_called_with_default(self, mock):
        _ = self.create([1, 2, 3])
        mock.assert_called_once_with(
            [1, 2, 3],
            dtype=None,
            device=None,
            requires_grad=False,
            pin_memory=False
        )

    @patch('torch.tensor')
    def test_tensor_called_with_custom(self, mock):
        device = pt.device('cpu')
        create = Create(pt.uint8, device, True, True)
        _ = create([1, 2, 3])
        mock.assert_called_once_with(
            [1, 2, 3],
            dtype=pt.uint8,
            device=device,
            requires_grad=True,
            pin_memory=True
        )

    def test_return_value_default(self):
        expected = pt.tensor([1, 2, 3])
        actual = self.create([1, 2, 3])
        pt.testing.assert_close(actual, expected)

    def test_return_value_custom(self):
        device = pt.device('cpu')
        create = Create(pt.float16, device, True, False)
        expected = pt.tensor([1, 2, 3], dtype=pt.float16, device='cpu')
        actual = create([1, 2, 3])
        pt.testing.assert_close(actual, expected)


class TestAsTensor(unittest.TestCase):

    def setUp(self):
        self.as_tensor = AsTensor()

    def test_default_dtype(self):
        self.assertTrue(hasattr(self.as_tensor, 'dtype'))
        self.assertIsNone(self.as_tensor.dtype)

    def test_default_device(self):
        self.assertTrue(hasattr(self.as_tensor, 'device'))
        self.assertIsNone(self.as_tensor.device)

    def test_custom_args(self):
        device = pt.device('cuda')
        as_tensor = AsTensor(pt.uint8, device)
        self.assertIs(as_tensor.dtype, pt.uint8)
        self.assertIs(as_tensor.device, device)

    def test_callable(self):
        self.assertTrue(callable(self.as_tensor))

    @patch('torch.as_tensor')
    def test_tensor_called(self, mock):
        _ = self.as_tensor([1, 2, 3])
        mock.assert_called_once()

    @patch('torch.as_tensor')
    def test_tensor_called_with_default(self, mock):
        _ = self.as_tensor([1, 2, 3])
        mock.assert_called_once_with([1, 2, 3], dtype=None, device=None)

    @patch('torch.as_tensor')
    def test_tensor_called_with_custom(self, mock):
        device = pt.device('cpu')
        as_tensor = AsTensor(pt.uint8, device)
        _ = as_tensor([1, 2, 3])
        mock.assert_called_once_with([1, 2, 3], dtype=pt.uint8, device=device)

    def test_return_value_default(self):
        expected = pt.tensor([1, 2, 3])
        actual = self.as_tensor([1, 2, 3])
        pt.testing.assert_close(actual, expected)

    def test_return_value_custom(self):
        device = pt.device('cpu')
        as_tensor = AsTensor(pt.float16, device)
        expected = pt.tensor([1, 2, 3], dtype=pt.float16, device='cpu')
        actual = as_tensor([1, 2, 3])
        pt.testing.assert_close(actual, expected)



class TestFromDataFrame(unittest.TestCase):

    def test_callable(self):
        self.assertTrue(callable(from_dataframe))

    @patch('torch.from_numpy')
    def test_from_numpy_called(self, mock):
        df = pd.DataFrame([1, 2, 3])
        _ = from_dataframe(df)
        mock.assert_called_once()

    @patch('torch.from_numpy')
    def test_from_numpy_called_with_values(self, mock):
        df = pd.DataFrame([1, 2, 3])
        _ = from_dataframe(df)
        np.testing.assert_array_equal(mock.call_args[0][0], df.values)

    def test_return_value(self):
        df = pd.DataFrame([1, 2, 3])
        actual = from_dataframe(df)
        pt.testing.assert_close(actual, pt.tensor(df.values, device='cpu'))



if __name__ == '__main__':
    unittest.main()
