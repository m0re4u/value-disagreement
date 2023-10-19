#!/bin/bash

set -ex

function gather {
    python3 scripts/value_profile_agreement.py --use_user_history --profile_min_sum $1 --profile_path $2 --subreddit $3 --similarity_method $4 --profile_processing $5 &> /dev/null &
}

function run_subreddit {
    # Value profiles
    gather 1 data/user_values_sum_normalized_bert.json $1 kendall sum_normalize
    gather 10 data/user_values_sum_normalized_bert.json $1 kendall sum_normalize
    gather 50 data/user_values_sum_normalized_bert.json $1 kendall sum_normalize
    gather 250 data/user_values_sum_normalized_bert.json $1 kendall sum_normalize
    gather 500 data/user_values_sum_normalized_bert.json $1 kendall sum_normalize


    gather 1 data/user_values_sum_normalized_bert.json $1 schwartz_soft_cosine sum_normalize
    gather 10 data/user_values_sum_normalized_bert.json $1 schwartz_soft_cosine sum_normalize
    gather 50 data/user_values_sum_normalized_bert.json $1 schwartz_soft_cosine sum_normalize
    gather 250 data/user_values_sum_normalized_bert.json $1 schwartz_soft_cosine sum_normalize
    gather 500 data/user_values_sum_normalized_bert.json $1 schwartz_soft_cosine sum_normalize

    gather 1 data/user_values_sum_normalized_bert.json $1 cosine_sim sum_normalize
    gather 10 data/user_values_sum_normalized_bert.json $1 cosine_sim sum_normalize
    gather 50 data/user_values_sum_normalized_bert.json $1 cosine_sim sum_normalize
    gather 250 data/user_values_sum_normalized_bert.json $1 cosine_sim sum_normalize
    gather 500 data/user_values_sum_normalized_bert.json $1 cosine_sim sum_normalize

    gather 1 data/user_values_sum_normalized_bert.json $1 absolute_error sum_normalize
    gather 10 data/user_values_sum_normalized_bert.json $1 absolute_error sum_normalize
    gather 50 data/user_values_sum_normalized_bert.json $1 absolute_error sum_normalize
    gather 250 data/user_values_sum_normalized_bert.json $1 absolute_error sum_normalize
    gather 500 data/user_values_sum_normalized_bert.json $1 absolute_error sum_normalize

    # Features with normalization
    # gather 1 data/user_features_minmax.json $1 kendall no_processing
    # gather 1 data/user_features_minmax.json $1 cosine_sim no_processing
    # gather 1 data/user_features_minmax.json $1 absolute_error no_processing

    # Centroids
    # gather 1 data/user_centroids_768.json $1 kendall no_processing
    # gather 1 data/user_centroids_768.json $1 cosine_sim no_processing
    # gather 1 data/user_centroids_768.json $1 absolute_error no_processing

    # # Noise
    # gather 1 data/user_noise.json $1 kendall no_processing
    # gather 1 data/user_noise.json $1 cosine_sim no_processing
    # gather 1 data/user_noise.json $1 absolute_error no_processing

    wait
    echo "Done with all scripts for $1"
}

run_subreddit climate
run_subreddit brexit
run_subreddit blacklivesmatter
run_subreddit republican
run_subreddit democrats