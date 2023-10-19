import json

import numpy as np

from value_disagreement.datasets import DebagreementDataset
from value_disagreement.extraction import ValueConstants, ValueDictionary
from value_disagreement.extraction.user_vectors import normalize_profile


dataset = DebagreementDataset("data/debagreement.csv", sample_N=-1)

LABEL_MAPPING = {
    0: "Disagree",
    1: "Neutral",
    2: "Agree"
}

def print_num_users(subset):
    df_subset = dataset.df[dataset.df.subreddit.str.lower() == subset.lower()]
    x = set(df_subset.author_child) | set(df_subset.author_parent)
    print(f"Subset {subset}: {len(set(x))}")
    return df_subset

print_num_users("brexit")
print_num_users("climate")
print_num_users("BlackLivesMatter")
print_num_users("democrats")
print_num_users("republican")


with open("data/user_values_sum_normalized_bert.json", "r") as f:
    user2vector = json.load(f)

dataset.filter_authors(user2vector.keys())

def print_value_ordering(subset):
    df_subset = dataset.df[dataset.df.subreddit.str.lower() == subset.lower()]
    users = list(df_subset['author_parent'])
    users.extend(df_subset['author_child'])
    users = set(users)
    users = [x for x in users if x in user2vector]
    # Normalize profiles
    vd = ValueDictionary(
        scoring_mechanism='any',
        aggregation_method=None,
        preprocessing=['lemmatize'],
    )

    all_profiles = np.array([normalize_profile(user2vector[x], "sum_normalize", vd=vd) for x in users])
    summed = np.mean(all_profiles, axis=0)
    # Print order of values
    reversed_sorted_list = np.argsort(summed)[::-1]
    print(f"Ordering of most important values for {subset}:")
    for i in reversed_sorted_list[:3]:
        print(f"  {ValueConstants.SCHWARTZ_VALUES[i]}: {summed[i]:.3f}")

def find_frequent_user_pairs(df):
    """
    Find parent/child pairs that are frequent in the dataset
    """
    print(df.columns)
    user_pairs = {}
    for i, row in df.iterrows():
        user_pairs[(row.author_parent, row.author_child)] = user_pairs.get((row.author_parent, row.author_child), 0) + 1
    user_pairs = {k: v for k, v in sorted(user_pairs.items(), key=lambda item: item[1], reverse=True)}
    print("Most frequent user pairs:")
    for i, (k, v) in enumerate(user_pairs.items()):
        if i > 10:
            break
        print(f"  {k}: {v}")
        # print interactions between these users
        print("    Interactions:")

        for i, row in df.iterrows():
            if row.author_parent == k[0] and row.author_child == k[1]:
                print(f"      ({row.subreddit}) {LABEL_MAPPING[row.label]}: {row.submission_text}")
                print(f"      [{row.author_parent}] {row.body_parent}")
                print(f"      ================> [{row.author_child}] {row.body_child}")


print("Num users after filtering")
print_num_users("brexit")
print_num_users("climate")
print_num_users("BlackLivesMatter")
print_num_users("democrats")
print_num_users("republican")

find_frequent_user_pairs(dataset.df)


print("Printing ordering of values")
print_value_ordering("brexit")
print_value_ordering("climate")
print_value_ordering("BlackLivesMatter")
print_value_ordering("democrats")
print_value_ordering("republican")


dataset = DebagreementDataset("data/debagreement.csv")
print("Number of items in this subset of Debagreement")
x = list(dataset.df['author_parent'])
x.extend(dataset.df['author_child'])
unique_users = set(x)
print(f"Number of unique users: {len(set(unique_users))}")
print("Problematic usernames: ")
for x in unique_users:
    if "/" in x:
        print(f"  {x}")