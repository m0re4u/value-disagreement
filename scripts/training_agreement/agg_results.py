import glob
import json

import pandas as pd


def print_results_for_table(results):
    df = pd.DataFrame(results)
    x = df[['test_precision', 'test_recall', 'test_f1', 'test_accuracy']].describe().round(2)
    relevant_columns = ['mean', 'std']
    df_results = x.loc[relevant_columns]
    for colname, y in df_results.items():
        print(f"{y['mean']:.2f}$_{{\\pm{y['std']:.2f}}}$", end=" & ")
    print()



single_model_names = [
    "agreement_trainer_bert-base-uncased_user_centroids_768.json",
    "agreement_trainer_bert-base-uncased_user_features_standard.json",
    "agreement_trainer_bert-base-uncased_user_noise.json",
    "agreement_trainer_bert-base-uncased_user_valueslemmatize_sum_normalized.json",
    "agreement_trainer_bert-base-uncased_nocontext",
    "agreement_trainer_bert-base-uncased_user_centroids_768",
    "agreement_trainer_bert-base-uncased_user_features_standard",
    "agreement_trainer_bert-base-uncased_user_noise",
    "agreement_trainer_bert-base-uncased_user_valueslemmatize_sum_normalized",
]

for model_name in single_model_names:
    metrics = []
    result_files = glob.glob(f"output/predictions/{model_name}*.json")
    context_only = False
    for file in result_files:
        with open(file) as f:
            data = json.load(f)
        if data['args']['only_context']:
            if "json" not in model_name:
                continue
            context_only = True
        metrics.append(data['metrics'])
    if context_only:
        print(f"{model_name} (context only)")
    else:
        print(f"model_name: {model_name}")
    print_results_for_table(metrics)
    print("-=======")
