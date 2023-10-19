import unittest

from value_disagreement.datasets import MFTCDataset


class TestMFTC(unittest.TestCase):
    def test_dataset(self):
        self.dataset = MFTCDataset()
        self.dataset.reformat_data()
        self.assertEqual(len(self.dataset), 34987)

