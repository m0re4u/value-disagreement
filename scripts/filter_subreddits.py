import random
from collections import Counter
import pandas as pd
from value_disagreement.datasets import RedditBackgroundDataset
import argparse


def filter_dataset(args):
    if args.subreddit == 'all':
        datasets = [
            RedditBackgroundDataset('brexit'),
            RedditBackgroundDataset('climate'),
            RedditBackgroundDataset('blacklivesmatter'),
            RedditBackgroundDataset('democrats'),
            RedditBackgroundDataset('republican'),
        ]
        user2data = {}
        user2metadata = {}
        for x in datasets:
            for u,y in x.user2data.items():
                if u in user2data:
                    user2data[u].extend(y)
                    user2metadata[u].extend(x.user2metadata[u])
                else:
                    user2data[u] = y
                    user2metadata[u] = x.user2metadata[u]
    else:
        dataset = RedditBackgroundDataset(args.subreddit)
        user2data = dataset.user2data
        user2metadata = dataset.user2metadata

    all_comments = []
    for user, comments in user2data.items():
        for i, comment in enumerate(comments):
            all_comments.append({
                "user": user,
                "text": comment,
                "id": i,
                "subreddit": user2metadata[user][i]['subreddit']
            })

    print(f"Number of users: {len(user2data)}")
    sr_set = set([x['subreddit'] for x in all_comments])

    with open('data/nsfw_reddit_list.txt') as f:
        nsfw_reddits = [x.strip().replace("/r/", "").lower() for x in f.readlines()]
    print(f"Number of NSFW subreddits: {len(nsfw_reddits)}")

    with open('data/gaming_reddit_list.txt') as f:
        gaming_reddits = [x.strip().lower() for x in f.readlines()]
    print(f"Number of gaming subreddits: {len(gaming_reddits)}")

    with open('data/image_reddit_list.txt') as f:
        image_reddits = [x.strip().lower() for x in f.readlines()]
    print(f"Number of image subreddits: {len(image_reddits)}")

    with open('data/left_over_list.txt') as f:
        left_over_reddits = [x.strip().lower() for x in f.readlines()]
    print(f"Number of extra subreddits ignored: {len(left_over_reddits)}")


    user_reddits = [x.lower() for x in sr_set if x.startswith("u_")]
    print(f"Number of user subreddits: {len(user_reddits)}")
    blacklists = nsfw_reddits + user_reddits + gaming_reddits + image_reddits + left_over_reddits
    print(f"Total number of blacklisted subreddits: {len(blacklists)} / {len(sr_set)}")

    filtered_comments = [x for x in all_comments if x['subreddit'].lower() not in blacklists]
    removed = len(all_comments) - len(filtered_comments)

    print(f"Raw number of comments: {len(all_comments)}")
    print(f"Num comments removed: {removed}")
    print(f"New number of comments: {len(filtered_comments)}")

    all_srs = [x['subreddit'] for x in filtered_comments]
    comment_counter = Counter(all_srs)

    srs_min_posts = [k for k,v in comment_counter.items() if v > 50]
    print(f"Number of subreddits kept: {len(srs_min_posts)}")
    new_filtered_comments = [x for x in filtered_comments if x['subreddit'] in srs_min_posts]
    print(f"Final number of comments: {len(new_filtered_comments)}")
    print(f"Final number of users: {len(set([x['user'] for x in new_filtered_comments]))}")

    df = pd.DataFrame.from_records(new_filtered_comments)
    df.to_csv(f"filtered_reddit_posts_{args.subreddit}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('subreddit', choices=['climate', 'blacklivesmatter','brexit', 'republican', 'democrats', 'all'],
                        help="which subreddit to filter")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    filter_dataset(args)
