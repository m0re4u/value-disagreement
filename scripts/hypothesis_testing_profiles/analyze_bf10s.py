import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def assign_bf_score(filename, split_on):
    """
    Based on the filename, assign a BF score to and extract the subreddit,
    similarity score, or threshold.
    """
    if split_on is None:
        return 'all'
    elif split_on == "subreddit":
        sr = filename.replace("output/test_results/test_user_values_sum_normalized_bert_sum_normalize_", "").split("_")[0]
        return sr
    elif split_on == "sim_score":
        for sim in ['schwartz_soft_cosine', 'kendall', 'cosine', 'absolute_error']:
            if sim in filename:
                return sim
    elif split_on == "threshold":
        if "500" in filename:
            return "500"
        elif "250" in filename:
            return "250"
        elif "50" in filename:
            return "50"
        elif "10" in filename:
            return "10"
        elif "1" in filename:
            return "1"
    else:
        raise ValueError("Unknown split_on value")

def plot_data(bfs):
    """
    Plot the BF test results.

    Also extract the right label mapping..
    """
    ax = sns.histplot(bfs, bins=30, multiple='stack', legend=True, palette="colorblind")
    color_mapping = {}
    for p, t in zip(ax.legend_.get_patches(), ax.legend_.texts):
        color_mapping[t._text] = p._facecolor
    color2text = {v: k for k, v in color_mapping.items()}
    label_colors = list(color_mapping.values())
    if len(label_colors) == 5:
        COLOR_TO_HATCH = {
            label_colors[0]: '//',
            label_colors[1]: 'x',
            label_colors[2]: 'o',
            label_colors[3]: '*',
            label_colors[4]: '+',
        }
    else:
        COLOR_TO_HATCH = {
            label_colors[0]: '//',
            label_colors[1]: 'x',
            label_colors[2]: 'o',
            label_colors[3]: '*',
        }

    added_labels = set()
    for i, bar in enumerate(ax.patches):
        if bar._facecolor in COLOR_TO_HATCH:
            bar.set_hatch(COLOR_TO_HATCH[bar._facecolor])
            if color2text[bar._facecolor] not in added_labels:
                added_labels.add(color2text[bar._facecolor])
                bar.set_label(color2text[bar._facecolor])

    ax.legend()
    container = ax.containers[0]
    dd = defaultdict(list)

    for p in container.patches:
        dd['x'].append(p.get_x() + p.get_width() / 2)

    idx = 0
    for container in ax.containers:
        ps = set([p.get_label() for p in container.patches])
        label = [x for x in ps if x != "_nolegend_"][0]
        for p in container.patches:
            dd[label].append(int(p.get_height()))
        idx += 1

    for key in dd.keys():
        print(f"{key} ", end="")
    print("")
    for i in range(len(dd['x'])):
        for key in dd.keys():
            print(f"{dd[key][i]} ", end="")
        print("")


def load_bfs(inpath, split):
    """
    Load the BF test results from json files, this is the output of the VPE +
    VPE experiments.
    """
    bfs = defaultdict(list)
    overview = []
    for filename in glob.glob(f'{inpath}/*.json'):
        with open(filename) as f:
            data = json.load(f)

        bf = float(data['BF'])
        if bf > 3:
            print(f"{bf:2.4f} - {filename}")
        overview.append((bf, filename))
        bf = min(bf, 30)
        bf_type = assign_bf_score(filename, split_on=split)
        bfs[bf_type].append(bf)
    top5 = list(sorted(overview, key=lambda x: x[0], reverse=True))[:5]
    print("Top 5")
    for score, fn in top5:
        print(f"{score} - {fn}")
    print("Min 5")
    min5 = list(sorted(overview, key=lambda x: x[0]))[:5]
    for score, fn in min5:
        print(f"{score} - {fn}")
    return bfs

def load_bfs_from_csv(infile, split):
    """
    Load the BF test results from a csv file, this is the output of the VPE +
    self report experiments.
    """
    bfs = defaultdict(list)
    overview = []
    hyp_df = pd.read_csv(infile, index_col=0)
    if split == "subreddit":
        for _, row in hyp_df.iterrows():
            bfs[row.subreddit.lower()].append(row.BF_kendall)
            bfs[row.subreddit.lower()].append(row.BF_absolute_error)
            bfs[row.subreddit.lower()].append(row.BF_cosine)
            bfs[row.subreddit.lower()].append(row.BF_soft_cosine)
    elif split == "sim_score":
        bfs['kendall'] = list(hyp_df.BF_kendall)
        bfs['absolute_error'] = list(hyp_df.BF_absolute_error)
        bfs['cosine'] = list(hyp_df.BF_cosine)
        bfs['schwartz_soft_cosine'] = list(hyp_df.BF_soft_cosine)
    else:
        bfs['all'] = list(hyp_df.BF_kendall) + list(hyp_df.BF_absolute_error) + list(hyp_df.BF_cosine) + list(hyp_df.BF_soft_cosine)
    overview = []
    for _, r in hyp_df.iterrows():
        overview.append((r.BF_kendall, f"Kendall + {r.subreddit}"))
        overview.append((r.BF_absolute_error, f"absoBF_absolute_error + {r.subreddit}"))
        overview.append((r.BF_cosine, f"cosine + {r.subreddit}"))
        overview.append((r.BF_soft_cosine, f"soft_cosine + {r.subreddit}"))
    top5 = list(sorted(overview, key=lambda x: x[0], reverse=True))[:5]
    print("Top 5")
    for score, fn in top5:
        print(f"{score} - {fn}")
    print("Min 5")
    top5 = list(sorted(overview, key=lambda x: x[0]))[:5]
    for score, fn in top5:
        print(f"{score} - {fn}")
    return bfs

def main(args):
    # some default values
    SCALE = 'log'
    Y_MAX = 13
    X_MAX = 25

    # Load data and parse depending on their source
    input_path = Path(args.input)
    if input_path.is_dir():
        bfs = load_bfs(input_path, args.split_on)
    else:
        bfs = load_bfs_from_csv(input_path, args.split_on)

    # Prepare figure
    plt.figure(figsize=(10, 5))
    plt.xlabel('BF score')
    plt.xscale(SCALE)
    plt.ylim(0, Y_MAX)
    plt.xlim(0.06, X_MAX)
    ax = plt.gca()
    rect_str_other = patches.Rectangle((0, 0), 0.05, Y_MAX, linewidth=0, fill=True, facecolor='g', edgecolor='r', alpha=0.2)
    rect_pos_other = patches.Rectangle((0.05, 0), 0.28, Y_MAX, linewidth=0, fill=True, facecolor='b', edgecolor='r', alpha=0.2)
    rect_non = patches.Rectangle((0.33, 0), 2.67, Y_MAX, linewidth=0, fill=True, facecolor='r', edgecolor='r', alpha=0.2)
    rect_pos = patches.Rectangle((3, 0), 17, Y_MAX, linewidth=0, fill=True, facecolor='b', edgecolor='r', alpha=0.2)
    rect_str = patches.Rectangle((20, 0), X_MAX, Y_MAX, linewidth=0, fill=True, facecolor='g', edgecolor='r', alpha=0.2)
    ax.add_patch(rect_str_other)
    ax.add_patch(rect_pos_other)
    ax.add_patch(rect_non)
    ax.add_patch(rect_pos)
    ax.add_patch(rect_str)

    # Add data
    plot_data(bfs)

    # Add hypothesis results
    plt.text(0.7, 11.4, 'H0 = H1', fontsize=9, fontstyle='italic')
    plt.text(0.1, 11.4, 'H0 > H1', fontsize=9, fontstyle='italic')
    plt.text(6, 11.4, 'H1 > H0', fontsize=9, fontstyle='italic')

    # Store figure
    print(f"Num tests: {len(bfs)}")
    plt.savefig(f'output/bf10s_{SCALE}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate results for the Bayes Factor tests")
    parser.add_argument('--split_on', type=str, default=None, help="Whether to split the results on a specific experiment variable")
    parser.add_argument('input', type=str, default="output/test_results", help="input path")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
