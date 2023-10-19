import unittest

import torch.nn as nn

from value_disagreement.extraction.agreement import dynamically_construct_mlp


class TestMLP(unittest.TestCase):
    def test_mlp_construction_1(self):
        mlp = dynamically_construct_mlp(2, 100, 50, 2)
        static_mlp = nn.Sequential(
            nn.Linear(100, 50),
            nn.GELU(),
            nn.Linear(50, 25),
            nn.GELU(),
            nn.Linear(25, 2)
        )

    def test_mlp_construction_2(self):
        mlp = dynamically_construct_mlp(5, 768, 300, 2)



if __name__ == '__main__':
    unittest.main()
