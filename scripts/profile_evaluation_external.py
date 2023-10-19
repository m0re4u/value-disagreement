import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from value_disagreement.extraction.user_vectors import (compute_absolute_error, compute_kendall,
                                     compute_soft_cos, compute_cos)

from value_disagreement.datasets import RedditBackgroundDataset
from value_disagreement.extraction import ValueDictionary


def compute_score(score_name, left, right):
    if score_name == 'kendall':
        score = compute_kendall(left, right)
    elif score_name == 'absolute_error':
        score = compute_absolute_error(left, right)
    elif score_name == 'soft_cosine':
        score = compute_soft_cos(left, right)
    elif score_name == 'cosine':
        score = compute_cos(left, right)
    return score

def print_ordered(u2u, profile_data, min_profile_sum=0.9, n=10, higher_is_more_similar=True):
    ind_x, ind_y = np.unravel_index(np.argsort(u2u, axis=None), u2u.shape)
    if higher_is_more_similar:
        ind_x = ind_x[::-1]
        ind_y = ind_y[::-1]
    j = 0

    user_triplets = []

    for i in range(20):
        profile_list = list(profile_data.items())
        user_names = [u for u, _ in profile_list]
        if np.sum(profile_list[ind_x[i]][1]) < min_profile_sum or np.sum(profile_list[ind_y[i]][1]) < min_profile_sum:
            print("Skipping user because profile sum is too low")
            continue
        print("-------")
        x_userid = ind_x[i]
        y_userid = ind_y[i]
        j += 1
        print("Similar users: ")
        print(f"X user: {profile_list[x_userid]}")
        x_username = profile_list[x_userid][0]
        print(f"Y user: {profile_list[y_userid]}")
        y_username = profile_list[y_userid][0]
        print(f"Similarity score between users: {u2u[x_userid, y_userid]}")

        # Looking for a third user
        other_users = list(range(len(profile_list)))
        dists = [u2u[x_userid, u] + u2u[y_userid, u] for u in other_users]
        sorted_dists_idx = np.argsort(dists)
        best_idx = 0 if higher_is_more_similar else -1
        third_user_id = sorted_dists_idx[best_idx]
        print(f"Third user id: {third_user_id} -> {profile_list[third_user_id]}")
        print(f"  Sim score to {x_username}: {u2u[x_userid, third_user_id]:.4f}")
        print(f"  Sim score to {y_username}: {u2u[y_userid, third_user_id]:.4f}")
        print(f"  Total score: {dists[third_user_id]:.4f}")
        user_triplets.append((x_username, y_username, user_names[third_user_id]))
        if j > n-1:
            break

    return user_triplets

def print_user_text(dataset, username, filtering="value_relevance", print_n=2):
    """
    Print the texts of a user.
    """
    vd = ValueDictionary(
        scoring_mechanism='any',
        aggregation_method=None,
        preprocessing='lemmatize'
    )
    print(dataset.head())
    user_rows = dataset[dataset.user == username]
    texts_to_print = []
    print(f"Username: {username}")
    print(f"Number of posts: {len(user_rows)}")
    print(f"Subreddits: {Counter(user_rows.subreddit)}")
    for _, row in user_rows.iterrows():
        text = row.text
        if filtering == "value_relevance":
            non_reply_text = "\n".join([t for t in text.split("\n") if len(t.strip()) and t.strip()[0] != ">" ])
            if vd.classify_comment_relevance(non_reply_text) == "y":
                texts_to_print.append(text)
        else:
            texts_to_print.append(text)

    print(f"Number of texts with values: {len(texts_to_print)}")
    device = torch.device("cuda")
    if torch.cuda.get_device_properties(0).total_memory > 10000000000:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")
        model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")
        inputs = torch.tensor([tokenizer.encode("\n".join(texts_to_print))])
        print(inputs)
        print(type(inputs))

        outputs = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)
        print("==== Summary ====")
        print(tokenizer.decode(outputs[0]))
        print("==== EOS ====")
    else:
        print("WARNING: GPU has less than 10GB of memory. This might cause problems.")

    texts_to_print = texts_to_print[:print_n]
    for t in texts_to_print:
        print("--")
        print(f" {t}")
    print("====================")


def load_all_background_data():
    """
    Load all Reddit background data.
    """
    reddit_brexit = RedditBackgroundDataset("brexit", load_csv="data/filtered_en_reddit_posts_brexit.csv")
    reddit_climate = RedditBackgroundDataset("climate", load_csv="data/filtered_en_reddit_posts_climate.csv")
    reddit_blm = RedditBackgroundDataset("blacklivesmatter", load_csv="data/filtered_en_reddit_posts_blacklivesmatter.csv")
    reddit_democrats = RedditBackgroundDataset("democrats", load_csv="data/filtered_en_reddit_posts_democrats.csv")
    reddit_republican = RedditBackgroundDataset("republican", load_csv="data/filtered_en_reddit_posts_republican.csv")
    concat_df = pd.concat([x.df for x in [reddit_brexit, reddit_climate, reddit_blm, reddit_democrats, reddit_republican]])
    return concat_df


def main(args):
    # Load args.profile_path from json
    with open(args.profile_path, 'r') as f:
        profile_data = json.load(f)

    profile_items = list(profile_data.items())
    profile_items = profile_items[:100]
    profile_data = {k: v for k, v in profile_items}

    n_profiles = len(profile_items)
    print(f"Number of profiles: {n_profiles}")
    u2u = np.zeros((n_profiles, n_profiles))
    for i, (_, p) in tqdm(enumerate(profile_items), total=n_profiles):
        for j, (_, other_p) in enumerate(profile_items[i+1:]):
            u2u[i,j] = compute_score(args.similarity_method, p, other_p)

    if args.similarity_method == 'kendall':
        user_triplets = print_ordered(u2u, profile_data, higher_is_more_similar=True, n=2)
    else:
        user_triplets = print_ordered(u2u, profile_data, higher_is_more_similar=False, n=2)

    print(user_triplets)
    user1, user2, user3 = user_triplets[0]

    # Load all background data
    background_data = load_all_background_data()

    # Print texts from three users
    print_user_text(background_data, user1)
    # print_user_text(background_data, user2)
    # print_user_text(background_data, user3)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract two similar and a distant user and show some stats (which ones??)")
    parser.add_argument('--profile_path', type=str, default="output/reddit_profiles_brexit_ValueEvalExtractor.json",
                        help="which profiles to load in case we do extra scraped data")
    parser.add_argument('--similarity_method', choices=["kendall", "absolute_error", "soft_cosine"],
                        required=True, default="kendall",
                        help="Which method to use for measuring profile overlap")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(args)
