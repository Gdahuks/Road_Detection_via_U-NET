import unittest
from model import *
import torch

class TestModel(unittest.TestCase):
    def test_same_size_0_mod_16(self):
        tensor = torch.randn((3, 1, 160, 160))
        model = UNET(in_channels=1, out_channels=1)
        preds = model(tensor)

        self.assertEqual(preds.shape, tensor.shape)

    def test_same_size_1_mod_16(self):
        tensor = torch.randn((3, 1, 161, 161))
        model = UNET(in_channels=1, out_channels=1)
        preds = model(tensor)

        self.assertEqual(preds.shape, tensor.shape)

    def test_same_size_15_mod_16(self):
        tensor = torch.randn((3, 1, 159, 159))
        model = UNET(in_channels=1, out_channels=1)
        preds = model(tensor)

        self.assertEqual(preds.shape, tensor.shape)

    def test_same_size_8_mod_16(self):
        tensor = torch.randn((3, 1, 152, 152))
        model = UNET(in_channels=1, out_channels=1)
        preds = model(tensor)

        self.assertEqual(preds.shape, tensor.shape)

    def test_diff_size_1(self):
        tensor = torch.randn((3, 1, 152, 160))
        model = UNET(in_channels=1, out_channels=1)
        preds = model(tensor)

        self.assertEqual(preds.shape, tensor.shape)

    def test_diff_size_2(self):
        tensor = torch.randn((3, 1, 172, 160))
        model = UNET(in_channels=1, out_channels=1)
        preds = model(tensor)

        self.assertEqual(preds.shape, tensor.shape)

    def test_diff_size_3(self):
        tensor = torch.randn((3, 1, 157, 160))
        model = UNET(in_channels=1, out_channels=1)
        preds = model(tensor)

        self.assertEqual(preds.shape, tensor.shape)

    def test_diff_size_4(self):
        tensor = torch.randn((3, 1, 157, 149))
        model = UNET(in_channels=1, out_channels=1)
        preds = model(tensor)

        self.assertEqual(preds.shape, tensor.shape)

if __name__ == '__main__':
    unittest.main()
