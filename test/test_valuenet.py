import unittest

from value_disagreement.datasets import ValueNetDataset


class TestValueNet(unittest.TestCase):
    def test_dataset(self):
        self.dataset = ValueNetDataset(data_dir="data/valuenet/")
        self.assertEqual(len(self.dataset), 28152)