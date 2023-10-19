import argparse
import json

import numpy as np
import pandas as pd

from value_disagreement.datasets import DebagreementDataset
from value_disagreement.extraction import ValueConstants
from value_disagreement.extraction.user_vectors import (compute_absolute_error,
                                             compute_cos, compute_kendall,
                                             compute_soft_cos)


def load_profiles(profile_path):
    """
    Load numerical profiles from a JSON file.
    """
    with open(profile_path, 'r') as f:
        profile_data = json.load(f)

    return profile_data


def insert_profiles(df, profiles):
    """
    Insert profiles into the dataframe.
    """
    parent = df.author_parent.isin(set(profiles.keys()))
    child = df.author_child.isin(set(profiles.keys()))
    df = df[parent & child]
    df["profile_parent"] = df.apply(lambda row: profiles[row["author_parent"]], axis=1)
    df["profile_child"] = df.apply(lambda row: profiles[row["author_child"]], axis=1)
    return df


def compute_similarities(df):
    df["kendall"] = df.apply(lambda row: compute_kendall(np.array(row["profile_parent"]), np.array(row["profile_child"])), axis=1)
    df["absolute_error"] = df.apply(lambda row: compute_absolute_error(np.array(row["profile_parent"]), np.array(row["profile_child"])), axis=1)
    df["soft_cosine"] = df.apply(lambda row: compute_soft_cos(np.array(row["profile_parent"]), np.array(row["profile_child"])), axis=1)
    df["cosine"] = df.apply(lambda row: compute_cos(np.array(row["profile_parent"]), np.array(row["profile_child"])), axis=1)
    return df


def print_instances(df, label, n_prints=20):
    df_agree = df[df.label == label]
    df_agree.sort_values(['kendall', 'absolute_error', 'soft_cosine'], ascending=[False, True, True], inplace=True)
    relevant_cols = ['body_parent', 'body_child', 'kendall', 'absolute_error', 'soft_cosine', 'cosine']
    with pd.option_context('display.max_colwidth', None,
                       'display.max_columns', None,
                       'display.max_rows', None):
        print(">>>> Most similar")
        for i in range(n_prints):
            print(df_agree.iloc[i][relevant_cols])
            print("=====================================")
        print(">>>> Least similar")
        for i in range(n_prints):
            print(df_agree.iloc[-(i+1)][relevant_cols])
            print("=====================================")


def print_avg_similarities(df, sr=None):
    df = df[df.subreddit == sr]
    print(f"Subreddit: {sr}")
    print(f"Average Kendall's tau: {df.kendall.mean()}")
    print(f"Average absolute error: {df.absolute_error.mean()}")
    print(f"Average soft cosine: {df.soft_cosine.mean()}")
    print(f"Average cosine: {df.cosine.mean()}")


def main(args):
    # Load data
    dataset = DebagreementDataset("data/debagreement.csv")
    profiles = load_profiles(args.profile_path)
    df = insert_profiles(dataset.df, profiles)
    df = compute_similarities(df)

    print_avg_similarities(df, sr="Brexit")
    print_avg_similarities(df, sr="climate")
    print_avg_similarities(df, sr="BlackLivesMatter")
    print_avg_similarities(df, sr="democrats")
    print_avg_similarities(df, sr="Republican")

    print("Agreeing instances ======================")
    print_instances(df, dataset.label2id['agree'])
    print("Disagreeing instances ======================")
    print_instances(df, dataset.label2id['disagree'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Examine text posts in Debagreement and cross reference with value profiles for conflict.")
    parser.add_argument('--profile_path', default=None, type=str,
                        help="Path to the profiles file to use.")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(args)
