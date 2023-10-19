import unittest

from value_disagreement.datasets import ValueEvalDataset


class TestValueEval(unittest.TestCase):
    def test_dataset(self):
        self.dataset = ValueEvalDataset(data_dir="data/valueeval/dataset-identifying-the-human-values-behind-arguments")
        self.assertEqual(len(self.dataset), 5270)