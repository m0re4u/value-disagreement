import unittest

from tqdm import tqdm

from value_disagreement.datasets import RedditBackgroundDataset
from value_disagreement.extraction import ValueDictionary
from value_disagreement.extraction.user_vectors import get_user2data


class TestComputeScores(unittest.TestCase):
    def test_vd(self):
        vd = ValueDictionary(
            scoring_mechanism='any',
            aggregation_method=None,
            preprocessing=['lemmatize']
        )

        reddit_brexit = RedditBackgroundDataset("brexit", load_csv="data/filtered_en_reddit_posts_brexit.csv")
        reddit_climate = RedditBackgroundDataset("climate", load_csv="data/filtered_en_reddit_posts_climate.csv")
        reddit_blm = RedditBackgroundDataset("blacklivesmatter", load_csv="data/filtered_en_reddit_posts_blacklivesmatter.csv")
        reddit_democrats = RedditBackgroundDataset("democrats", load_csv="data/filtered_en_reddit_posts_democrats.csv")
        reddit_republican = RedditBackgroundDataset("republican", load_csv="data/filtered_en_reddit_posts_republican.csv")
        subreddit_data = [reddit_brexit, reddit_climate, reddit_blm, reddit_democrats, reddit_republican]
        user2data = get_user2data(subreddit_data)
        i = 0

        keys = list(user2data.keys())[:10]
        small_u2d = {k: list(user2data[k])[:10] for k in keys}
        for user in tqdm(keys, total=len(keys)):
            for comment in small_u2d[user]:
                labels = vd.classify_comment_value(comment)
                if len(labels) > 0:
                    i += 1
            if i == 100:
                break

        pp = vd.profile_multiple_users(small_u2d)
