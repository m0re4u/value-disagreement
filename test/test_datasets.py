import datetime
import glob
import json
import unittest

from value_disagreement.datasets import *
from value_disagreement.extraction import ValueConstants
from value_disagreement.extraction.user_vectors import get_user2data

constant_mapping = {i: k for i, k in enumerate(ValueConstants.SCHWARTZ_VALUES)}


class TestDatasets(unittest.TestCase):
    def test_valuenet(self):
        self.valuenet = ValueNetDataset(
            "data/valuenet/", return_predefined_splits=True)
        self.assertEqual(len(self.valuenet), 28152)
        train_idx, _, _ = self.valuenet.get_splits()
        first_train_item = self.valuenet[train_idx].iloc[0]
        self.assertEqual(first_train_item['text'], " After accomplishing every task I cross each item off my list.")
        self.assertEqual(first_train_item['value'], "power")
        self.assertEqual(first_train_item['orig_label'], 1)



    def test_valueeval(self):
        self.valueeval = ValueEvalDataset(
            "data/valueeval/dataset-identifying-the-human-values-behind-arguments/",
            cast_to_valuenet=True,
            return_predefined_splits=True
        )
        self.assertEqual(len(self.valueeval), 55090)
        train_idx, _, _ = self.valueeval.get_splits()
        first_train_item = self.valueeval[train_idx].iloc[0]
        self.assertEqual(first_train_item['text'], "if entrapment can serve to more easily capture wanted criminals, then why shouldn't it be legal?")
        self.assertEqual(first_train_item['value'], "security")
        self.assertEqual(first_train_item['orig_label'], 1)

    def test_reddit_background(self):
        reddit_brexit = RedditBackgroundDataset("brexit", load_csv="data/filtered_en_reddit_posts_brexit.csv")
        reddit_climate = RedditBackgroundDataset("climate", load_csv="data/filtered_en_reddit_posts_climate.csv")
        reddit_blm = RedditBackgroundDataset("blacklivesmatter", load_csv="data/filtered_en_reddit_posts_blacklivesmatter.csv")
        reddit_democrats = RedditBackgroundDataset("democrats", load_csv="data/filtered_en_reddit_posts_democrats.csv")
        reddit_republican = RedditBackgroundDataset("republican", load_csv="data/filtered_en_reddit_posts_republican.csv")
        subreddit_data = [reddit_brexit, reddit_climate, reddit_blm, reddit_democrats, reddit_republican]
        self.user2data = get_user2data(subreddit_data)
