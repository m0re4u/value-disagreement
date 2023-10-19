import argparse

import fasttext

from value_disagreement.datasets import RedditBackgroundDataset

"""
You may need to download additional files.

1. Download model from [here](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz).
2. Place in top-level project directory.
3. Run `scripts/filter_reddit.py` for every subreddit.

"""


def main(args):
    # Load data
    dataset = RedditBackgroundDataset(args.subreddit, load_csv=f"data/filtered_reddit_posts_{args.subreddit}.csv")
    orig_len = len(dataset.df)
    # Load language detection model
    model = fasttext.load_model('lid.176.ftz')

    df = dataset.df
    df['lang'] = df['text'].apply(lambda x: model.predict(str(x).replace('\n', ''), k=1)[0][0])

    # Filter out non-English
    out_df = df[df['lang'] == "__label__en"]
    print(f"Rows dropped: {orig_len} - {len(out_df)} = {orig_len - len(out_df)}")
    out_df = out_df.reset_index(drop=True)
    out_df.to_csv(f"data/filtered_en_reddit_posts_{args.subreddit}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Filter data scraped from reddit based on detected language (should be English).")
    parser.add_argument('--subreddit', choices=["climate", "brexit", "blacklivesmatter", "republican", "democrats"], default=None, type=str,
                        help="Which subcorpus to load")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(args)
