import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pingouin import ttest
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from value_disagreement.extraction import ValueConstants
from value_disagreement.extraction.user_vectors import (compute_absolute_error,
                                             compute_cos, compute_kendall,
                                             compute_soft_cos)

def load_pve_profiles(pve_profiles_path, reapply_scaling=True):
    """
    Load a json file with PVE profiles
    """
    with open(pve_profiles_path, 'r') as f:
        pve_profiles = json.load(f)

    scaler = MinMaxScaler()
    if reapply_scaling:
        for user, profile in pve_profiles.items():
            pve_profiles[user] = scaler.fit_transform(np.array(profile).reshape(-1, 1)).reshape(-1)
    return pve_profiles


def load_pvq_profiles(pvq_profiles_dir_path):
    """
    Load a json file per user and return a dictionary with the profiles
    """
    pvq_profiles = {}
    pp = Path(pvq_profiles_dir_path)
    for file in pp.glob("*.json"):
        with open(file, 'r') as f:
            user = file.stem.split("=")[1]
            user_dict = json.load(f)
            profile = [user_dict['centered_value_scores'][value] for value in ValueConstants.SCHWARTZ_VALUES]
            scaler = MinMaxScaler()
            profile = scaler.fit_transform(np.array(profile).reshape(-1, 1)).reshape(-1)
            pvq_profiles[user] = profile
    return pvq_profiles


def compute_similarity_scores(df, sim_score="kendall", reapply_scaling=True):
    """
    Compute similarity scores between parent and child profiles.
    """
    if sim_score == "kendall":
        sim_func = compute_kendall
    elif sim_score == "absolute_error":
        sim_func = compute_absolute_error
    elif sim_score == "cosine":
        sim_func = compute_cos
    elif sim_score == "soft_cosine":
        sim_func = compute_soft_cos

    return df.apply(lambda x: sim_func(x['parent_profile'], x['child_profile']), axis=1)


def run_ttests_and_draw(data, print_text="", filename="human_hypothesis_testing.png", also_output_plot_data=True):
    """
    Run a statistical test on the data containing profile similarity scores.

    Also draw a boxplot for each similarity score.
    """
    SIMILARITY_SCORES = ['kendall', 'absolute_error', 'cosine', 'soft_cosine']
    df_agree = data[data.accept == 'agree']
    df_disagree = data[data.accept == 'disagree']
    if len(df_disagree) < 2 or len(df_agree) < 2:
        print(f"Skipping task hash because one of the groups is too small")
        return None

    # create four subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    # Storage for the scores for outputting to latex
    scores = defaultdict(dict)

    # Record test results to later analyze
    tests = {}
    for i, (sim, tailed) in enumerate(zip(SIMILARITY_SCORES, ['less', 'greater', 'less', 'less'])):
        # Compute t-test for each similarity score
        test_results = ttest(df_disagree[sim], df_agree[sim], alternative=tailed, paired=True, r=0.5)
        bf_score = float(test_results.BF10)
        tests[sim] = test_results

        this_ax = axs[i // 2, i % 2]
        data.boxplot(column=sim, by='accept', ax=this_ax)
        this_ax.title.set_text(f"{sim} (tailed: {tailed}) - BF10: {bf_score:.3f}")
        scores[sim]['agree'] = df_agree[sim]
        scores[sim]['disagree'] = df_disagree[sim]

    if also_output_plot_data:
        agree_data = "\\\\\n".join([f"{x:.5f}" for x in scores['absolute_error']['agree'].to_list()])
        disagree_data = "\\\\\n".join([f"{x:.5f}" for x in scores['absolute_error']['disagree'].to_list()])
        print(
"""
% agree
\\addplot
table[row sep=\\\\,y index=0, mark=none] {
    data\\\\"""
        )
        print(agree_data, end="\\\\ \n")
        print("};")

        print(
"""
% disagree
\\addplot
table[row sep=\\\\,y index=0, mark=none] {
    data\\\\"""
        )
        print(disagree_data, end="\\\\ \n")
        print("};")

    fig.suptitle(print_text, fontsize=12)
    plt.savefig(Path("output") / filename)
    plt.show()
    return tests


def select_diverse_annotators(pvq_profiles):
    """
    Select those profiles that are the most distinct from each other
    """
    scores = {
        'kendall': [],
        'absolute_error': [],
        'cosine': [],
        'soft_cosine': [],
    }
    distances = np.zeros((len(pvq_profiles.keys()),len(pvq_profiles.keys()),4))
    user_pairs = list(combinations(pvq_profiles.keys(), 2))
    user2index = {user: i for i, user in enumerate(pvq_profiles.keys())}
    index2user = {i: user for i, user in enumerate(pvq_profiles.keys())}
    for user1, user2 in tqdm(user_pairs):
        distances[user2index[user1], user2index[user2], 0] = compute_kendall(pvq_profiles[user1], pvq_profiles[user2])
        distances[user2index[user1], user2index[user2], 1] = compute_absolute_error(pvq_profiles[user1], pvq_profiles[user2])
        distances[user2index[user1], user2index[user2], 2] = compute_cos(pvq_profiles[user1], pvq_profiles[user2])
        distances[user2index[user1], user2index[user2], 3] = compute_soft_cos(pvq_profiles[user1], pvq_profiles[user2])

    for user in pvq_profiles.keys():
        scores['kendall'].append(np.mean(distances[user2index[user], :, 0]))
        scores['absolute_error'].append(np.mean(distances[user2index[user], :, 1]))
        scores['cosine'].append(np.mean(distances[user2index[user], :, 2]))
        scores['soft_cosine'].append(np.mean(distances[user2index[user], :, 3]))


    # Remove users based on their similarity scores
    users = []
    for sim in scores.keys():
        print(f"Top 10 users for {sim}")
        if sim in ['kendall', 'cosine', 'soft_cosine']:
            argsorted_scores = np.argsort(scores[sim])[::-1]
        else:
            argsorted_scores = np.argsort(scores[sim])
        sorted_top_scores = argsorted_scores[:7]
        for user_idx in sorted_top_scores:
            print(f"{index2user[user_idx]}->{scores[sim][user_idx]:.4f}")
            users.append(index2user[user_idx])
    user_counter = Counter(users)
    keep_profiles = {user: pvq_profiles[user] for user in pvq_profiles if user_counter[user] < 3}
    print(f"Keeping {len(keep_profiles)} profiles")
    return keep_profiles


def print_average_profiles_sim(estimated_profiles, self_reported_profiles, method="kendall"):
    """
    print the average similarity between all combinations of estimated profiles and self-reported profiles
    """
    scores = []
    for user in tqdm(estimated_profiles.keys()):
        for user2 in self_reported_profiles.keys():
            if method == "kendall":
                scores.append(compute_kendall(estimated_profiles[user], self_reported_profiles[user2]))
            elif method == "absolute_error":
                scores.append(compute_absolute_error(estimated_profiles[user], self_reported_profiles[user2]))
            elif method == "cosine":
                scores.append(compute_cos(estimated_profiles[user], self_reported_profiles[user2]))
            elif method == "soft_cosine":
                scores.append(compute_soft_cos(estimated_profiles[user], self_reported_profiles[user2]))
            else:
                raise ValueError(f"Unknown method {method}")
    print(f"Average {method} score: {np.mean(scores):.4f}")


def main(args):
    json_list = []
    outfile_suffix = ""
    for file in args.input_files:
        with open(file, 'r') as json_file:
            json_list.extend([json.loads(x) for x in list(json_file)])
    print(f"Loaded {len(json_list)} samples from {len(args.input_files)} files")
    df = pd.DataFrame.from_records(json_list)

    # Load user value profiles
    PVE_PROFILES = load_pve_profiles(args.pve_profiles)
    PVQ_PROFILES = load_pvq_profiles(args.pvq_profiles)
    df['parent_profile'] = df['author_parent'].apply(lambda x: np.array(PVE_PROFILES[x]))
    df['prolific_ids'] = df['_annotator_id'].apply(lambda x: x.replace("debagreement_small-", ""))
    df['child_profile'] = df['prolific_ids'].apply(lambda x: np.array(PVQ_PROFILES[x]))
    if args.select_diverse_annotators:
        outfile_suffix += "_diverse"
        loaded_profiles = [x for x in df['prolific_ids'].unique()]
        print(loaded_profiles)
        PVQ_PROFILES = select_diverse_annotators({x: np.array(PVQ_PROFILES[x]) for x in loaded_profiles})
        old_len = len(df)
        df = df[df.prolific_ids.isin(PVQ_PROFILES.keys())]
        print(f"Removed {old_len - len(df)} samples because annotator was not diverse enough")


    # Compute all four similarity scores
    df['kendall'] = compute_similarity_scores(df, sim_score="kendall")
    df['absolute_error'] = compute_similarity_scores(df, sim_score="absolute_error")
    df['cosine'] = compute_similarity_scores(df, sim_score="cosine")
    df['soft_cosine'] = compute_similarity_scores(df, sim_score="soft_cosine")

    # Separate on given label from user
    df.accept = df.accept.apply(lambda x: x[0])

    if args.split_on == "record":
        task_instances = []
        for record, df_record in df.groupby("_task_hash"):
            print(f"Running hypothesis testing on {record}")
            tests = run_ttests_and_draw(
                df_record,
                print_text=f"{df_record.iloc[0].submission_text} -> {df_record.iloc[0].body_parent}",
                filename=f"human_hypothesis_testing_{record}{outfile_suffix}.png",
                also_output_plot_data=args.print_plot_data)
            if tests is None:
                continue

            task_instances.append({
                "task_hash": record,
                "subreddit": df_record.iloc[0].subreddit,
                "count_agree": len(df_record[df_record.accept == 'agree']),
                "count_disagree": len(df_record[df_record.accept == 'disagree']),
                "submission_text": df_record.iloc[0].submission_text,
                "body_parent": df_record.iloc[0].body_parent,
                "BF_kendall": float(tests['kendall']['BF10']),
                "BF_absolute_error": float(tests['absolute_error']['BF10']),
                "BF_cosine": float(tests['cosine']['BF10']),
                "BF_soft_cosine": float(tests['soft_cosine']['BF10']),
                "pval_kendall": float(tests['kendall']['p-val']),
                "pval_absolute_error": float(tests['absolute_error']['p-val']),
                "pval_cosine": float(tests['cosine']['p-val']),
                "pval_soft_cosine": float(tests['soft_cosine']['p-val']),
            })
        print(f"Number of task instances: {len(task_instances)}")
        # Create dataframe from tests
        df_tests = pd.DataFrame.from_records(task_instances)
        df_tests.to_csv("output/analyze_hypothesis_testing.csv")
    elif args.split_on == "subreddit":
        for subreddit, df_subreddit in df.groupby("subreddit"):
            # print average similaity scores for each subreddit
            print(f"Running hypothesis testing on {subreddit}")
            print_average_profiles_sim(df_subreddit.parent_profile, df_subreddit.child_profile, method="kendall")
            print_average_profiles_sim(df_subreddit.parent_profile, df_subreddit.child_profile, method="absolute_error")
            print_average_profiles_sim(df_subreddit.parent_profile, df_subreddit.child_profile, method="cosine")
            print_average_profiles_sim(df_subreddit.parent_profile, df_subreddit.child_profile, method="soft_cosine")
            run_ttests_and_draw(
                df_subreddit,
                print_text=f"{subreddit}",
                filename=f"human_hypothesis_testing_{subreddit}{outfile_suffix}.png",
                also_output_plot_data=args.print_plot_data)
    elif args.split_on == "user":
        for user, df_user in df.groupby("_annotator_id"):
            print(f"Running hypothesis testing on {user}")
            run_ttests_and_draw(
                df_user,
                print_text=f"{user}",
                filename=f"human_hypothesis_testing_{user}{outfile_suffix}.png",
                also_output_plot_data=args.print_plot_data)
    else:
        run_ttests_and_draw(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple jsonl with annotations files into one dataset to be annotated")
    parser.add_argument('--input_files', type=str, nargs="+", default=None, help="input annotation files")
    parser.add_argument('--pvq_profiles', type=str, default=None, help="input directory for profiles generated by PVQ method")
    parser.add_argument('--pve_profiles', type=str, default=None, help="input path for profiles generated by PVE method")
    parser.add_argument('--enable_filter', action="store_true", default=False, help="turn on a default filter which will exclude some instances depending on the split_on argument. Only works for split_on=record.")
    parser.add_argument('--select_diverse_annotators', action="store_true", default=False, help="Make a selection of diverse annotators based on their profiles. ")
    parser.add_argument('--print_plot_data', action="store_true", default=False, help="turn on printing of data to plot in paper.")
    parser.add_argument('--split_on', type=str, default=None, choices=['record', 'subreddit', 'user'], help="by which data field to split the annotations on.")
    args = parser.parse_args()
    main(args)


