import json
import unittest
from itertools import combinations
from pathlib import Path


class TestProfiles(unittest.TestCase):
    def test_user_set(self):
        self.files = [
            'data/user_centroids_768.json',
            'data/user_centroids_prefiltered.json',
            'data/user_features_minmax.json',
            'data/user_features_standard.json',
            'data/user_noise.json',
            'data/user_values_sum_normalized_bert.json',
            'data/user_valueslemmatize_freqweight_normalized.json',
            'data/user_valueslemmatize_sum_normalized.json',
        ]

        user_sets = []
        for file in self.files:
            if not Path(file).exists():
                continue
            with open(file) as f:
                data = json.load(f)
                self.assertEqual(len(data.keys()), 18548)
                user_sets.append(set(data.keys()))
        for set1, set2 in combinations(user_sets, 2):
            self.assertEqual(set1, set2)



if __name__ == '__main__':
    unittest.main()