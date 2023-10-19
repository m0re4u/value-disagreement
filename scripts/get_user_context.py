import argparse
from value_disagreement.datasets import DebagreementDataset, RedditBackgroundDataset
from value_disagreement.extraction.user_vectors  import get_centroids, get_user_features, get_values, get_noise
from itertools import permutations

USER_CONTEXT_TYPES = [
    'centroid',
    'features',
    'features_stdscaled',
    'values',
    'values_normed',
    'values_freqnormed', # also lemmatized
    'values_lemmatized',
    'values_bert',
    'noise',
    'centroid_prefiltered'
]


def get_overlap_in_users(subreddit_datas):
    for x,y in permutations(subreddit_datas, 2):
        base_users = set(x.user2data.keys())
        other_users = set(y.user2data.keys())
        print(f"{x.subreddit} -> {y.subreddit}: {len(base_users.intersection(other_users))}")


def main(args):
    # Load all datasets
    deba = DebagreementDataset("data/debagreement.csv")
    reddit_brexit = RedditBackgroundDataset("brexit", load_csv="data/filtered_en_reddit_posts_brexit.csv")
    reddit_climate = RedditBackgroundDataset("climate", load_csv="data/filtered_en_reddit_posts_climate.csv")
    reddit_blm = RedditBackgroundDataset("blacklivesmatter", load_csv="data/filtered_en_reddit_posts_blacklivesmatter.csv")
    reddit_democrats = RedditBackgroundDataset("democrats", load_csv="data/filtered_en_reddit_posts_democrats.csv")
    reddit_republican = RedditBackgroundDataset("republican", load_csv="data/filtered_en_reddit_posts_republican.csv")

    # Get all debagreement texts
    corpus_parent = [x for x,_,_,_ in deba]
    corpus_child = [y for _,y,_,_ in deba]
    all_texts = corpus_parent + corpus_child

    # Combine data from all subreddits
    subreddit_data = [reddit_brexit, reddit_climate, reddit_blm, reddit_democrats, reddit_republican]

    # Optionally, print the statustics for user overlaps
    if args.print_stats:
        get_overlap_in_users(subreddit_data)
        for sr in subreddit_data:
            print(f"{sr.subreddit} - Found: {len(sr.user2data):4} Comments: {len(sr)}")

    # Get the profile vectors depending on the method to use
    if args.method == 'centroid':
        get_centroids(subreddit_data, all_texts)
    elif args.method == 'centroid_prefiltered':
        get_centroids(subreddit_data, all_texts, prefilter=True)
    elif args.method == 'features':
        get_user_features(subreddit_data, preprocessing="minmax")
    elif args.method == 'features_stdscaled':
        get_user_features(subreddit_data, preprocessing="standard")
    elif args.method == 'values':
        get_values(subreddit_data)
    elif args.method == 'values_normed':
        get_values(subreddit_data,
            aggregation_method='sum_normalized'
        )
    elif args.method == 'values_freqnormed':
        get_values(subreddit_data,
            preprocessing=['lemmatize'],
            aggregation_method='freqweight_normalized'
        )
    elif args.method == 'values_lemmatized':
        get_values(subreddit_data,
            preprocessing=['lemmatize'],
            aggregation_method='sum_normalized'
        )
    elif args.method == 'values_bert':
        get_values(subreddit_data,
            preprocessing=[],
            aggregation_method='sum_normalized',
            value_estimation='trained_bert',
            checkpoint="moral_values_trainer_bert-base-uncased_both_1/"
        )
    elif args.method == 'noise':
        get_noise(subreddit_data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create vectors for each user using any of the methods described.")
    parser.add_argument('method', choices=USER_CONTEXT_TYPES,
                    help="which method to use for creating user vectors")
    parser.add_argument('--print_stats', action='store_true',
                    help="If True, only print stats for the subreddits")

    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(args)


