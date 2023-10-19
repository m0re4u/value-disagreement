import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pingouin import ttest

from value_disagreement.datasets import DebagreementDataset
from value_disagreement.extraction import AutoValueExtractor, ValueDictionary
from value_disagreement.extraction.user_vectors import (compute_absolute_error,
                                             compute_cos, compute_kendall,
                                             compute_soft_cos)
from value_disagreement.vizualization.author_counts import plot_subreddit_author_counts
from value_disagreement.vizualization.profile_similarities import (
    box_plot_correlations, get_output_filename, plot_profile_matrix,
    violin_plot_correlations)


def get_profile_method_from_filename(filename):
    """
    Get the name of the method used to create the user profiles based on
    filename.
    """
    if "user_values" in filename:
        return Path(filename).stem
    elif "user_features" in filename:
        return "features"
    elif "user_centroids" in filename:
        return "centroids"
    elif "noise" in filename:
        return "noise"
    else:
        raise ValueError(f"Unknown profile method for filename: {filename}")

def get_all_profiles(sr_authors, model_path, profile_path, use_user_history=True):
    """
    Get value profiles of users in the dataset.

    If use_user_history is True, then use the extracted Reddit data. Otherwise, use the posts
    inside the debagreement data.
    """
    if not use_user_history:
        value_extractor = AutoValueExtractor.create_extractor(model_path)
        # Get profiles
        user2profile = value_extractor.profile_multiple_users(sr_authors)
        profiling_method = "values"
    else:
        # Load in profile mapping
        with open(profile_path) as f:
            stored_data = json.load(f)

        profiling_method = get_profile_method_from_filename(profile_path)

        if 'profiles' in stored_data:
            # old style loading?
            user2profile = {k: np.array(v) for k, v in stored_data['profiles'].items()}
        else:
            user2profile = {k: np.array(v) for k, v in stored_data.items()}
        print(f"Number of users with history data: {len(user2profile)}")

        user2profile = {k: v for k, v in user2profile.items()
                        if k in sr_authors}
        print(f"Filtered to only contain subreddit authors: {len(user2profile)}")
        first_profile = list(user2profile.values())[0]

        for author in [k for k in sr_authors if k not in user2profile]:
            user2profile[author] =  np.expand_dims(np.zeros_like(first_profile), axis=0)

        if "used_model_path" in stored_data:
            if "value_dictionary" in stored_data['used_model_path']:
                vd = ValueDictionary()
                id2label = vd.id2label
            else:
                value_extractor = AutoValueExtractor.create_extractor(stored_data['used_model_path'])
                id2label = value_extractor.trainer.model.config.id2label
        else:
            id2label = None

    return profiling_method, id2label, user2profile


def filter_profiles(profiles, profile_min_sum=1, profile_processing=None, plot=False):
    """
    Filter value profile based on the sum of the value counts.
    """
    nonzero_profiles = {}
    ignored_profiles = 0
    vd = ValueDictionary(
            scoring_mechanism='any',
            aggregation_method='freqweight_normalized',
            preprocessing=['lemmatize'],
    )
    sums = []
    for user in profiles:
        profile = profiles[user]
        sums.append(np.sum(profile))
        if not np.sum(profile) >= profile_min_sum:
            ignored_profiles += 1
        else:
            if profile_processing == "freqweight":
                prof = profile / vd.word_freq_weight
                profile = vd.sum_normalize_profile(prof)
            elif profile_processing == "sum_normalize":
                profile = vd.sum_normalize_profile(profile)
            nonzero_profiles[user] = profile
    print(
        f"Num profiles with only zeros (ignored): {ignored_profiles}/{len(profiles)} = {ignored_profiles/len(profiles)}")
    if plot:
        plt.figure()
        sns.distplot(sums)
        plt.savefig("output/profile_sums.png")
    return nonzero_profiles


def get_profile_agreement_correlation(df, profiles, id2label, sim_fn, exclude_seen_pairs=False):
    """
    Get the similarity scores between authors of parent and child posts.

    if exclude_seen_pairs is True, only include each pair of authors once.
    """
    label_mapping = {
        0: 'DISAGREE',
        1: 'NEUTRAL',
        2: 'AGREE'
    }

    entries = {'items': [], 'id2label': id2label}
    label2sims = {
        "DISAGREE": [],
        "NEUTRAL": [],
        "AGREE": [],
    }
    if exclude_seen_pairs:
        seen_pairs = set()
    for _, row in df.iterrows():
        if row['author_parent'] in profiles and row['author_child'] in profiles:
            if exclude_seen_pairs:
                if (row['author_parent'], row['author_child']) in seen_pairs:
                    continue
            sim = sim_fn(profiles[row['author_parent']], profiles[row['author_child']])
            label2sims[label_mapping[row['label']]].append(sim)
            x = {
                "text_parent": row['body_parent'],
                "text_child": row['body_child'],
                "profile_parent": profiles[row['author_parent']].tolist(),
                "profile_child": profiles[row['author_child']].tolist(),
                "similarity_score": sim,
                "label": label_mapping[row['label']]
            }
            entries['items'].append(x)
            if exclude_seen_pairs:
                seen_pairs.add((row['author_parent'], row['author_child']))
                seen_pairs.add((row['author_child'], row['author_parent']))

    num_corrs = sum([len(v) for _, v in label2sims.items()])
    print(f"Number of correlation scores over all labels: {num_corrs}")
    return label2sims, entries


def get_subreddit_authors(df, subreddit, plot=False):
    # Select relevant subreddit
    subreddit_df = df[df.subreddit.str.lower() == subreddit]

    # Get list of all authors
    x = list(subreddit_df['author_parent'])
    x.extend(subreddit_df['author_child'])
    cc = Counter(x)
    print(f"Number of {subreddit} authors: {len(cc)} - Total number of messages in {subreddit} data: {sum([v for k, v in cc.items()])}")
    if plot:
        plot_subreddit_author_counts(cc, subreddit)
    return set(cc.keys())


def main(args):
    if args.similarity_method == "kendall":
        sim_fn = compute_kendall
        tailed = "less"
    elif args.similarity_method == "absolute_error":
        sim_fn = compute_absolute_error
        tailed = "greater"
    elif args.similarity_method == "cosine_sim":
        sim_fn = compute_cos
        tailed = "less"
    elif args.similarity_method == "schwartz_soft_cosine":
        sim_fn = compute_soft_cos
        tailed = "less"
    else:
        raise ValueError("Unknown value profile similarity function")

    dataset = DebagreementDataset("data/debagreement.csv")

    # Obtain number of comments per author
    subreddit_authors = get_subreddit_authors(dataset.df, args.subreddit, args.plot_author_counts)

    # Get profiles for users
    profile_method, id2label, profiles = get_all_profiles(
        subreddit_authors,
        model_path=args.model_path,
        profile_path=args.profile_path,
        use_user_history=args.use_user_history,
    )
    profiles = filter_profiles(profiles, args.profile_min_sum, args.profile_processing)

    # Create heatmap with user-user value profile correlation
    if args.plot_profile_matrix:
        plot_profile_matrix(profiles, sim_fn=sim_fn)

    # Create value profile correlation distribution split on agreement-label
    label2sims, entries = get_profile_agreement_correlation(
        dataset.df,
        profiles,
        id2label,
        sim_fn=sim_fn,
        exclude_seen_pairs=args.exclude_seen_pairs,
    )

    test_results = ttest(label2sims['DISAGREE'], label2sims['AGREE'], alternative=tailed, r=0.5)
    print(test_results)

    args.BF = test_results.BF10['T-test']
    args.profiling_method = profile_method
    test_out_filename = get_output_filename(args, 'test', prefix='output/test_results/', filetype='json')
    output_dict = vars(args)

    with open(test_out_filename, 'w') as f:
        json.dump(output_dict, f)

    if args.plot_correlations:
        box_plot_correlations(label2sims, args)
        violin_plot_correlations(label2sims, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_user_history', action="store_true",
                        help="Instead of profiling users based on DEBAGREEMENT data, use the extra scraped data")
    parser.add_argument('--exclude_seen_pairs', action="store_true",
                        help="Only parse every parent/child pair once per test")
    parser.add_argument('--profile_min_sum', type=int, default=1,
                        help="Minimum sum of the value counts in the user profile")
    parser.add_argument('--model_path', type=str, default=None,
                        help="which model to use for profiling in case we do DEBAGREEMENT")
    parser.add_argument('--profile_processing', type=str, default=None,
                        help="Add any extra processing to profile vectors")
    parser.add_argument('--similarity_method', choices=["kendall", "absolute_error", "cosine_sim", "schwartz_soft_cosine"],
                        required=True, default="kendall",
                        help="Which method to use for measuring profile overlap")
    parser.add_argument('--subreddit', choices=["climate", "brexit", "blacklivesmatter", "democrats", "republican"],
                        required=True, default="climate",
                        help="Which subreddit to analyze for")
    parser.add_argument('--profile_path', type=str, default=None,
                        help="which profiles to load in case we do extra scraped data")
    parser.add_argument('--plot_profile_matrix', action='store_true',
                        help="Enable plotting the user-user profile matrix")
    parser.add_argument('--plot_author_counts', action='store_true',
                        help="Enable plotting the subreddit author comment counts")
    parser.add_argument('--plot_correlations', action='store_true',
                        help="Enable plotting of box and violin plot for profile similarity scores per label")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(args)
