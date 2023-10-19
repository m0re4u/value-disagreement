import argparse
import json
from collections import defaultdict
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import MDS, TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from value_disagreement.datasets import RedditBackgroundDataset
from value_disagreement.extraction import ValueConstants, ValueDictionary
from value_disagreement.extraction.user_vectors import normalize_profiles


def preprocess(texts, embedding_method):
    if embedding_method == "tfidf":
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer())
        ])
        X = pipeline.fit_transform(texts).toarray()
    elif embedding_method == "longformer":
        pass
    return X

def get_vd(filter_method):
    if "min_count" in filter_method:
        min_value_count = int(filter_method.split("_")[-1])
        vd = ValueDictionary(
            scoring_mechanism='min_count',
            aggregation_method=None,
            preprocessing='lemmatize',
            min_value_count=min_value_count
        )
    elif filter_method == "value_dictionary":
        vd = ValueDictionary(
            scoring_mechanism='any',
            aggregation_method=None,
            preprocessing='lemmatize'
        )
    return vd


def prefilter(texts, filter_method):
    vd = get_vd(filter_method)
    value_comments = []
    for text in texts:
        judgement = vd.classify_comment_value(text)
        if len(judgement):
            value_comments.append(text)
    return value_comments


def plot_tsne(z, y, outname="output/tsne_profiles.png", print_data=True):
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    if len(y) == 10:
        df["value_text"] = ValueConstants.SCHWARTZ_VALUES
    elif len(y) == 4:
        df["value_text"] = ValueConstants.SCHWARTZ_HIGHER_ORDER
    plt.figure()
    fig, ax = plt.subplots(figsize=(10,7))
    fig.tight_layout(rect=[0,0,.8,1])
    g = sns.scatterplot(
        x="comp-1",
        y="comp-2",
        s=500,
        hue=df.y.tolist(),
        palette=sns.color_palette("hls", len(set(y))),
        data=df,
        ax=ax,
    )
    if print_data:
        for i in range(len(z)):
            print(f"{z[i, 0]:.6f} {z[i, 1]:.6f} {df['value_text'].iloc[i]} {ValueConstants.SCHWARTZ_HIGHER_ORDER_MAPPING[df['value_text'].iloc[i]]}")
    g.set(title="2D projection")
    g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
    for i, row in df.iterrows():
        plt.annotate(ValueConstants.SCHWARTZ_VALUES[i], (row["comp-1"], row["comp-2"]))
    print(f"Outputting figure to: {outname}")
    plt.savefig(outname)


def profile_clustering(profiles, method="kmeans"):
    """
    Cluster users on their profile
    """
    users = profiles.keys()
    if method == "kmeans":
        num_clusters = 10
        X = np.stack(profiles.values())
        kms = KMeans(n_clusters=num_clusters, random_state=0)
        cluster_assignment = kms.fit_predict(X)
    elif method == "top1":
        cluster_assignment = np.array([np.argmax(profile) for profile in profiles.values()])
    elif method == "schwartz_higher":
        cluster_assignment = []
        for user in profiles:
            higher_order_count = {
                'self-enhancement': 0,
                'self-transcendence': 0,
                'conservation': 0,
                'openness-to-change': 0,
            }
            for i, value in enumerate(ValueConstants.SCHWARTZ_VALUES):
                higher_order_count[ValueConstants.SCHWARTZ_HIGHER_ORDER_MAPPING[value]] += profiles[user][i]
            assignment = max(higher_order_count, key=higher_order_count.get)
            cluster_assignment.append(ValueConstants.SCHWARTZ_HIGHER_ORDER.index(assignment))
        cluster_assignment = np.array(cluster_assignment)

    # for i in range(num_clusters):
    #     print(f"Cluster {i}: {len(cluster_assignment[cluster_assignment == i])}")

    return list(users), cluster_assignment

def scale_profiles(profiles, filter_entropy=0.0):
    X = np.stack(profiles.values())
    if filter_entropy > 0:
        X = X[entropy(X, axis=1) > filter_entropy, :]
    X_standard = StandardScaler().fit_transform(X)
    return X_standard

def estimate_covariance(profiles):
    X_scaled = scale_profiles(profiles)
    cov = EmpiricalCovariance(assume_centered=True).fit(X_scaled)
    return cov

def plot_covariance(cov, collapse_profiles=False, outname="output/covariance.png"):
    plt.figure()
    ax = plt.gca()
    ax.imshow(cov.covariance_)
    if collapse_profiles:
        ax.set_xticks(np.arange(len(ValueConstants.SCHWARTZ_HIGHER_ORDER)), labels=ValueConstants.SCHWARTZ_HIGHER_ORDER)
        ax.set_yticks(np.arange(len(ValueConstants.SCHWARTZ_HIGHER_ORDER)), labels=ValueConstants.SCHWARTZ_HIGHER_ORDER)
    else:
        ax.set_xticks(np.arange(len(ValueConstants.SCHWARTZ_VALUES)), labels=ValueConstants.SCHWARTZ_VALUES)
        ax.set_yticks(np.arange(len(ValueConstants.SCHWARTZ_VALUES)), labels=ValueConstants.SCHWARTZ_VALUES)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    plt.savefig(outname)

def plot_cov_mds(
        cov,
        collapse_profiles=False,
        algo="MDS",
        metric=True,
        random_state=1,
        n_init=25,
        outname="output/covariance_decomp.png"):
    if algo == "mds":
        model = MDS(n_components=2, metric=metric, dissimilarity='precomputed', random_state=random_state, verbose=1, max_iter=1000, n_jobs=8, n_init=n_init, eps=1e-5)
    elif algo == "pca":
        model = PCA(n_components=3, random_state=1, svd_solver='full')
    elif algo == "tsne":
        model = TSNE(n_components=2, random_state=1, verbose=1, n_jobs=8, n_iter=1000, perplexity=30)
    z = model.fit_transform(cov.covariance_)

    # Set label colors depending on Schwartz circumplex
    if collapse_profiles:
        higher_order_sch = ValueConstants.SCHWARTZ_HIGHER_ORDER
    else:
        higher_order_sch = [ValueConstants.SCHWARTZ_HIGHER_ORDER_MAPPING[x] for x in ValueConstants.SCHWARTZ_VALUES]

    outname = outname.replace(".png", f"_{algo}.png")
    plot_tsne(z, higher_order_sch, outname=outname)

def collapse_profiles(profiles):
    new_profiles = {}
    for user in profiles:
        new_profiles[user] = np.zeros(4)
        for j, v in enumerate(ValueConstants.SCHWARTZ_VALUES):
            idx = ValueConstants.SCHWARTZ_HIGHER_ORDER.index(ValueConstants.SCHWARTZ_HIGHER_ORDER_MAPPING[v])
            new_profiles[user][idx] += profiles[user][j]
    return new_profiles

def main(args):
    with open(args.profile_path, 'r') as f:
        profiles = json.load(f)

    if args.collapse_profiles:
        profiles = collapse_profiles(profiles)

    if args.profile_processing is not None:
        profiles = normalize_profiles(profiles, args.profile_processing)

    if args.analysis == "covariance":
        cov = estimate_covariance(profiles)
        plot_covariance(cov, args.collapse_profiles)
        plot_cov_mds(cov, args.collapse_profiles, algo=args.algo)
    else:
        # cluster profiles
        users, cass = profile_clustering(profiles, method=args.profile_cluster_method)

        # subsample users to make figure more readable
        subsample = 1000
        user_idx = sample(range(len(users)), subsample)
        users = [users[i] for i in user_idx]
        cass = [cass[i] for i in user_idx]

        # Load text posts for all groups
        print("Loading text data...")
        reddit_brexit = RedditBackgroundDataset("brexit", load_csv="data/filtered_reddit_posts_brexit.csv")
        reddit_climate = RedditBackgroundDataset("climate", load_csv="data/filtered_reddit_posts_climate.csv")
        reddit_blm = RedditBackgroundDataset("blacklivesmatter", load_csv="data/filtered_reddit_posts_blacklivesmatter.csv")
        reddit_democrats = RedditBackgroundDataset("democrats", load_csv="data/filtered_reddit_posts_democrats.csv")
        reddit_republican = RedditBackgroundDataset("republican", load_csv="data/filtered_reddit_posts_republican.csv")
        all_sr_data = [reddit_brexit, reddit_climate, reddit_blm, reddit_democrats, reddit_republican]

        # Matching texts across subreddits to users
        user2doc = defaultdict(list)
        for user in tqdm(users):
            for sr in all_sr_data:
                if user in sr.user2data:
                    if args.text_prefilter_method is not None:
                        user2doc[user].extend(prefilter(sr.user2data[user], args.text_prefilter_method))
                    else:
                        user2doc[user].extend([str(x) for x in sr.user2data[user]])

        all_texts = []
        nonzero_users = []
        nonzero_cass = []
        for i, user in enumerate(users):
            if len(user2doc[user]):
                all_texts.append(" ".join(user2doc[user]))
                nonzero_users.append(user)
                nonzero_cass.append(cass[i])

        cass = nonzero_cass
        users = nonzero_users

        print("Running preprocessing...")
        # T-SNE on text data
        X = preprocess(all_texts, args.embedding)
        for i, user in enumerate(users):
            print(f"user {user}: {len(user2doc[user])} - {np.sum(X[i,:]):2.2f}")
        print(X.shape)


        print(f"Doing {args.algo}...")
        if args.algo == "tsne":
            model = TSNE(n_components=2, random_state=0, verbose=1)
        elif args.algo == 'mds':
            model = MDS(n_components=2, random_state=0, verbose=1)

        z = model.fit_transform(X)
        # plot_tsne(z, y, outname="output/tsne_profiles.png")
        experiment_name = f"manifold_profiles_{args.profile_cluster_method}_{args.algo}_{args.text_prefilter_method}"
        plot_tsne(z, cass, outname=f"output/{experiment_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform an analysis based on the loaded profiles. Includes MDS, T-SNE, K-means etc.")
    parser.add_argument('--profile_path', type=str, default=None),
    parser.add_argument('--analysis', type=str, default="covariance", help="Which analysis to perform. Options: covariance, manifold"),
    parser.add_argument('--collapse_profiles', action='store_true', default=False, help="Collapse profiles to 4 higher order schwartz dimensions"),
    parser.add_argument('--algo', type=str, default='mds', choices=['mds', 'tsne', 'pca'], help="Which algorithm to use for the covariance analysis"),
    parser.add_argument('--text_embedding', type=str, default=None),
    parser.add_argument('--text_prefilter_method', type=str, default=None),
    parser.add_argument('--profile_cluster_method', type=str, default="kmeans"),
    parser.add_argument('--profile_processing', type=str, default=None, help="Whether to add any processing to the profiles. Options: None, sum_normalize, freqweight"),
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(args)


