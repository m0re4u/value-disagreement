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

def write_redditor_posts(user, output_data_dir):
    reddit_user = reddit.redditor(user)
    new_comments = reddit_user.comments.new(limit=None)
    outfile = output_data_dir / f"{user}.json"

    comment_data = []
    for comment in new_comments:
        comment_data.append({
            'subreddit': comment.subreddit.display_name,
            'body': comment.body,
            'score': comment.score,
            'created_utc': comment.created_utc,
            'id': comment.id,
            'permalink': comment.permalink,
        })
    with open(outfile, "w" ) as f:
        json.dump(comment_data, f)
    print(f"Wrote {len(comment_data)} comments to file for user {user}")


def scrape(args):
    OUT_FOLDER = Path(f"output/reddit_data_{args.subreddit}/")
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    OUT_FOLDER_IGNORED = Path(f"output/reddit_data_ignoredusers/")
    OUT_FOLDER_IGNORED.mkdir(parents=True, exist_ok=True)
    dataset = DebagreementDataset("data/debagreement.csv", sample_N=-1)
    df = dataset.df
    subreddit_df = df[df.subreddit.str.lower() == args.subreddit]
    print("Number of items in this subset of Debagreement")
    x = list(subreddit_df['author_parent'])
    x.extend(subreddit_df['author_child'])
    unique_users = set(x)

    print(f"Number of users in {args.subreddit} data: {len(unique_users)}")
    for user in tqdm.tqdm(unique_users):
        try:
            if (OUT_FOLDER / f"{user}.json").exists() or (OUT_FOLDER_IGNORED / f"{user}.json").exists():
                print(f"{user}.json already exists, skipping")
                continue
            write_redditor_posts(user, OUT_FOLDER)
        except (prawcore.exceptions.NotFound, prawcore.exceptions.Forbidden, prawcore.exceptions.BadRequest) as e:
            with open(OUT_FOLDER_IGNORED / f"{user}.json", "w") as f:
                json.dump({"error": str(e)}, f)
            print(f"User '{user}' not found (reaon {e})")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape text posts from provided Reddit usernames.")
    parser.add_argument('subreddit', choices=['climate', 'blacklivesmatter','brexit', 'democrats', 'republican'],
                        help="which subreddit to scrape")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    scrape(args)
