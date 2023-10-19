import argparse
import json
import os
from pathlib import Path

import praw
import prawcore
import tqdm
from dotenv import load_dotenv

from value_disagreement.datasets import DebagreementDataset

load_dotenv()

reddit = praw.Reddit(
    client_id=os.environ["PRAW_CLIENT_ID"],
    client_secret=os.environ["PRAW_API_KEY"],
    user_agent="python:vbda:v0.0.1",
)

def get_num_giled(user):
    """
    Returns the number of posts made by the user that have been gilded
    """
    total = sum([1 for _ in user.gilded()])
    return total

def get_num_comments(user):
    return sum([1 for _ in user.comments.new(limit=None)])

def get_num_submissions(user):
    return sum([1 for _ in user.submissions.new(limit=None)])


def get_redditor_metrics(user):
    reddit_user = reddit.redditor(user)

    num_gilded = get_num_giled(reddit_user)
    num_comments = get_num_comments(reddit_user)
    num_submissions = get_num_submissions(reddit_user)
    return {
        'username': user,
        'comment_karma': reddit_user.comment_karma,
        'link_karma': reddit_user.link_karma,
        'created_utc': reddit_user.created_utc,
        'is_gold': reddit_user.is_gold,
        'is_mod': reddit_user.is_mod,
        'is_employee': reddit_user.is_employee,
        'num_gilded': num_gilded,
        'num_comments': num_comments,
        'num_submissions': num_submissions,
    }

def write_redditor_features(user, features, out_folder):
    outfile = out_folder / f"{user}.json"
    with open(outfile, "w" ) as f:
        json.dump(features, f)
    print(f"Wrote features to file for user {user}")

def main(args):
    OUT_FOLDER = Path(f"output/reddit_userfeatures/")
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    OUT_FOLDER_IGNORED = Path(f"output/reddit_userfeatures_ignoredusers/")
    OUT_FOLDER_IGNORED.mkdir(parents=True, exist_ok=True)
    dataset = DebagreementDataset("data/debagreement.csv")
    print("Number of items in this subset of Debagreement")
    x = list(dataset.df['author_parent'])
    x.extend(dataset.df['author_child'])
    unique_users = set(x)
    print(f"Number of users in data: {len(unique_users)}")
    for user in tqdm.tqdm(unique_users, total=len(unique_users)):
        try:
            if (OUT_FOLDER / f"{user}.json").exists() or (OUT_FOLDER_IGNORED / f"{user}.json").exists():
                print(f"{user}.json already exists, skipping")
                continue
            features = get_redditor_metrics(user)
            write_redditor_features(user, features, OUT_FOLDER)
        except (prawcore.exceptions.NotFound, prawcore.exceptions.Forbidden, prawcore.exceptions.BadRequest) as e:
            with open(OUT_FOLDER_IGNORED / f"{user}.json", "w") as f:
                json.dump({"error": str(e)}, f)
            print(f"User '{user}' not found (reaon {e})")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Reddit user features given some reddit usernames")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
