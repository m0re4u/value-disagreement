import json

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from value_disagreement.datasets import DebagreementDataset
from value_disagreement.utils import print_stats


def create_df(dataset, user_context=None):
    stemmer = SnowballStemmer(language='english')
    records = []
    for x, y, z, extra_info in dataset:
        parent = extra_info['author_parent']
        child = extra_info['author_child']
        if user_context is not None:
            if parent not in user_context or child not in user_context:
                continue
            records.append({
                'text': stemmer.stem(f"{x}. {y}"),
                'label': z,
                'user_left': parent,
                'user_right': child,
            })
        else:
            records.append({
                'text': stemmer.stem(f"{x}. {y}"),
                'label': z,
            })
    return pd.DataFrame.from_records(records)


def load_user_context(user_context_path):
    with open(user_context_path, 'r') as f:
        user_context = json.load(f)
    return user_context


def preprocess(df, user_context=None):
    pp = Pipeline([
        ('vect', CountVectorizer(max_features=10000)),
        ('tfidf', TfidfTransformer()),
     ])
    X = pp.fit_transform(df.text).toarray()
    if user_context is not None:
        authors_left_X = np.stack([np.array(user_context[x]) for x in df.user_left])
        authors_right_X = np.stack([np.array(user_context[x]) for x in df.user_left])
        X = np.concatenate([X, authors_left_X, authors_right_X], axis=1)
    y = df.label
    return X, y


def create_model_and_space(model_type):
    if model_type == 'svm':
        param_space = {
            'C': np.logspace(-6, 6, 30),
            'gamma': np.logspace(-8, 8, 30),
            'tol': np.logspace(-4, -1, 30)
        }
        model = SVC(kernel='rbf', verbose=1, C=0.25, tol=0.001, class_weight='balanced')
    elif model_type == 'logreg':
        param_space = {
            'C': np.logspace(-6, 6, 30),
            'tol': np.logspace(-5, 0, 30),
            'max_iter': np.logspace(3, 5, 10)
        }
        model = LogisticRegression(verbose=1, C=0.09, max_iter=5000, tol=0.3, class_weight='balanced', solver='saga')
    elif model_type == 'all_ones':
        model = DummyClassifier(strategy='most_frequent', random_state=42)
        param_space = None
    return model, param_space


def get_results(model, X, y, object_fn=False):
    predicted = model.predict(X)
    return metrics.classification_report(y, predicted, digits=3, output_dict=object_fn)


def main(args):
    deba = DebagreementDataset("data/debagreement.csv")
    if args.user_context is not None:
        uc = load_user_context(args.user_context)
    else:
        uc = None
    df = create_df(deba, uc)
    if args.only_subsample:
        uc = None
    X, y = preprocess(df, uc)
    X_train, X_int, y_train, y_int = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_int, y_int, test_size=0.5, random_state=42)
    print(f"Train size: {X_train.shape} - val_size: {X_val.shape} - test_size: {X_test.shape}")

    model, param_space = create_model_and_space(args.model_type)
    if args.ray:
        search = RandomizedSearchCV(model, param_space, scoring='f1_macro', n_jobs=-1, cv=2, verbose=0, n_iter=50)
        import joblib
        from ray.util.joblib import register_ray
        register_ray()
        with joblib.parallel_backend('ray'):
            search.fit(X_train, y_train)
        print(search.best_params_)
        print(get_results(search.best_estimator_, X_test, y_test))
    else:
        reses = []
        for i in range(args.n_seeds):
            print(f"Seed: {i}")
            model, param_space = create_model_and_space(args.model_type)
            model.fit(X_train, y_train)
            res = get_results(model, X_test, y_test, object_fn=True)
            reses.append(res)
        print_stats("Precision", [res['macro avg']['precision'] for res in reses])
        print_stats("Recall", [res['macro avg']['recall'] for res in reses])
        print_stats("F1", [res['macro avg']['f1-score'] for res in reses])
        print_stats("Accuracy", [res['accuracy'] for res in reses])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split', type=str, default='temporal', choices=['random', 'temporal'],
                        help="how to split the debagreement data into sets")
    parser.add_argument('--ray', default=False, action='store_true',
                        help="If true, use ray for distributed hyperparameter search.")
    parser.add_argument('--n_seeds', default=1, type=int,
                        help="if >1, do multiple runs and average the results (+stdev)")
    parser.add_argument('--model_type', type=str, default='svm', choices=['svm', 'logreg', 'all_ones'],
                        help="which model to use")
    parser.add_argument('--user_context', type=str, default=None,
                        help="If defined, use this path for loading user context vectors")
    parser.add_argument('--only_subsample', action='store_true',
                        help="If True, only subsample the data (user context vectors are provided, but not used in the model)")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
