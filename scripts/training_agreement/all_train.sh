#!/bin/bash

set -e
# Only context
python3 scripts/training_agreement/train.py --only_context --learning_rate 0.001 --num_epochs 10 --user_context data/user_noise.json --n_runs 5
python3 scripts/training_agreement/train.py --only_context --learning_rate 0.001 --num_epochs 10 --user_context data/user_centroids_768.json --n_runs 5
python3 scripts/training_agreement/train.py --only_context --learning_rate 0.001 --num_epochs 10 --user_context data/user_features_standard.json --n_runs 5
python3 scripts/training_agreement/train.py --only_context --learning_rate 0.001 --num_epochs 10 --user_context data/user_valueslemmatize_sum_normalized.json --n_runs 5


# TF-IDF + Logreg
python3 scripts/training_agreement/baseline.py --model_type logreg --user_context data/user_features_standard.json --only_subsample --n_seeds 5
# + epsilon
python3 scripts/training_agreement/baseline.py --model_type logreg --user_context data/user_noise.json --n_seeds 5
# + z
python3 scripts/training_agreement/baseline.py --model_type logreg --user_context data/user_centroids_768.json --n_seeds 5
# + u
python3 scripts/training_agreement/baseline.py --model_type logreg --user_context data/user_features_standard.json --n_seeds 5
# + v
python3 scripts/training_agreement/baseline.py --model_type logreg --user_context data/user_valueslemmatize_sum_normalized.json --n_seeds 5


# BERT
python3 scripts/training_agreement/train.py --use_model bert-base-uncased --num_epochs 10 --learning_rate 5e-05 --user_context data/user_features_standard.json --only_subsample --n_runs 5
# + epsilon
python3 scripts/training_agreement/train.py --use_model bert-base-uncased --num_epochs 10 --learning_rate 5e-05 --user_context data/user_noise.json --n_runs 5
# + z
python3 scripts/training_agreement/train.py --use_model bert-base-uncased --num_epochs 10 --learning_rate 5e-05 --user_context data/user_centroids_768.json --n_runs 5
# + u
python3 scripts/training_agreement/train.py --use_model bert-base-uncased --num_epochs 10 --learning_rate 5e-05 --user_context data/user_features_standard.json --n_runs 5
# + v
python3 scripts/training_agreement/train.py --use_model bert-base-uncased --num_epochs 10 --learning_rate 5e-05 --user_context data/user_valueslemmatize_sum_normalized.json --n_runs 5
