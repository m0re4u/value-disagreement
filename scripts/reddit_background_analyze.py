import argparse
import json

import matplotlib.pyplot as plt
import seaborn as sns
from value_disagreement.datasets import RedditBackgroundDataset
from value_disagreement.extraction import AutoValueExtractor, ValueDictionary


def main(args):
    dataset = RedditBackgroundDataset(args.subreddit, load_csv=f"data/filtered_reddit_posts_{args.subreddit}.csv")
    print(f"Total number of users: {len(dataset.user2data)}")
    print(f"Total number of comments available: {len(dataset)}")

    lengths = [len(str(post)) for user in dataset.user2data for post in dataset.user2data[user] if len(str(post)) < 1000]
    plt.figure()
    sns.displot(lengths)
    plt.savefig(f"output/reddit_comment_lengths_{args.subreddit}.png")

    if args.profile_history:
        if args.use_model == "value_dictionary":
            ve = ValueDictionary()
        else:
            ve = AutoValueExtractor.create_extractor(args.use_model)
        # Extract a profile for users based on all their comments
        user2profile = ve.profile_multiple_users(dataset.user2data)

        # Save profiles to file
        dump_profiles = {k:v.tolist() for k,v in user2profile.items()}
        dump_dict = {
            'used_model_path': args.use_model,
            'used_model_name': ve.__class__.__name__,
            'subreddit': args.subreddit,
            'profiles': dump_profiles
        }
        with open(f"output/reddit_profiles_{args.subreddit}_{ve.__class__.__name__}.json", "w") as f:
            json.dump(dump_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load background Reddit data and perform profiling.")
    parser.add_argument('--profile_history', action="store_true", default=False,
                        help="Perform profiling users based on gathered reddit history")
    parser.add_argument('--subreddit', choices=["climate", "brexit", "blacklivesmatter", "republican", "democrats"], default="climate", type=str,
                        help="Which subcorpus to load")
    parser.add_argument('--use_model', default="scripts/training_moral_values/valueeval_trainer/checkpoint-792/",
                        help="Which model to use for the value predictor")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(args)
