import argparse

import numpy as np
from value_disagreement.datasets import RedditAnnotatedDataset, ValueEvalDataset, ValueNetDataset
from value_disagreement.extraction import ValueDictionary
from sklearn.metrics import classification_report


def get_dataset(dataset_name):
    """
    Load dataset based on name.
    """
    if dataset_name == "valueeval":
        dataset = ValueEvalDataset(
            "~/projects/implicit/aspects/data/valueeval/dataset-identifying-the-human-values-behind-arguments/",
            cast_to_valuenet=True,
            return_predefined_splits=True
        )
    elif dataset_name == "valuenet":
        dataset = ValueNetDataset("~/projects/implicit/aspects/data/valuenet/", return_predefined_splits=True)
    else:
        dataset = RedditAnnotatedDataset(dataset_name)
    return dataset

def get_results(dataset, value_dictionary, test_idx):
    """
    Get predictions using the value dictionary.
    """
    y_true = []
    y_pred_vd = []
    for i in test_idx:
        label = dataset[i]['orig_label']
        text = dataset[i]['text']
        dictionary_pred = value_dictionary.classify_comment_value(text)
        value = dataset[i]['value']
        y_pred_vd.append(int(value in dictionary_pred))
        y_true.append(abs(label))

    return y_pred_vd, y_true


def print_results(y_true, y_pred, test_idx):
    """
    Print the results.
    """
    print("Value dictionary")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("All ones")
    y_pred_all_ones = np.ones((len(test_idx)))
    print(classification_report(y_true, y_pred_all_ones, zero_division=0))

def main(args):
    value_dictionary = ValueDictionary(scoring_mechanism="any")

    if args.dataset == 'both':
        dataset_eval = get_dataset("valueeval")
        dataset_net = get_dataset("valuenet")
        _, _, test_ve_idx = dataset_eval.get_splits()
        _, _, test_vn_idx = dataset_net.get_splits()

        y_pred_ve, y_true_ve = get_results(dataset_eval, value_dictionary, test_ve_idx)
        y_pred_vn, y_true_vn = get_results(dataset_net, value_dictionary, test_vn_idx)

        y_pred = np.concatenate((y_pred_ve, y_pred_vn))
        y_true = np.concatenate((y_true_ve, y_true_vn))
        test_idx = np.concatenate((test_ve_idx, test_vn_idx))
    else:
        dataset = get_dataset(args.dataset)
        print("Dataset size:", len(dataset))
        _, _, test_idx = dataset.get_splits()
        print("Test size:", len(test_idx))
        y_pred, y_true = get_results(dataset, value_dictionary, test_idx)

    print_results(y_true, y_pred, test_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help="which annotated dataset to load")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(args)

