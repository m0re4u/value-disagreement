# Value Disagreements
Code accompanying the EMNLP2023 long paper "Do Differences in Values Influence Disagreements in Online Discussions?"


## Installation
1. Install the package in developer mode `pip install -e .`. This will (hopefully) also install the dependencies.
2. Done ... (well, almost)

You may need to download or initialize models, but you will be prompted for it.
- You may need to download spacy models
- You may beed to initialize wandb logging

## Folder structure
Here's some guidance on the folder structure. Inside each file should be more information about what the script is intended to do.

### value_disagreements
The code for the value extraction, dataset and evaluation metrics. Some classes require external files, listed below:
- Value Dictionary baseline: find the Refined_dictionary.txt file [here](https://osf.io/vt8nf/) and place it in `data/`.

### Notebooks
May contain notebooks made for analysis of generated or scraped data. In our case, contains code for training the TF-IDF baseline for (dis-)agreement prediction.

### Data
Folder for storing all data (datasets, user profile information, task instances, survey results). We list a bunch of sources below.
- Experimental data for our paper: https://osf.io/42dns/
- Debagreement: https://scale.com/open-av-datasets/oxford
- ArgValues (internal name is ValueEval): https://zenodo.org/records/6855004 (download `webis-argvalues-22.zip`).
- ValueNet: https://liang-qiu.github.io/ValueNet/ (download original).
- MFTC: https://osf.io/k5n7y/ (download and follow provided instructions)

### Test
_Some_ unittest functionality, or other sanity checks. Call using `python3 -m unittest discover test`.

### Scripts
#### Hypothesis testing profiles
Creation of the Bayes Factor scores.
#### Training agreement
Training and evaluation of models for agreement analysis.
#### Training Moral Values
Training and evaluation of models for value extractions

## Constructing user profiles
Download the experimental data from OSF, which contains the links to the Reddit comments analyzed in our work. You can gather these comments using e.g. [PRAW](https://praw.readthedocs.io/en/stable/). After obtaining the comment data, you can construct user profiles as follows.

1. Filter the comments to only include data from relevant subreddits with `scripts/filter_subreddits.py`. You may need to adjust internal paths and the comment storage format to match that of the `RedditBackgroundDataset`.
2. Filter the content to only include English text using `scripts/filter_reddit.py`.
3. Create user profiles using `scripts/get_user_context.py`. Depending on the method you are using for constructing the profiles, you may need to have trained value extraction models (see next section).


## Training models
### Training Value Extraction models
See `python3 scripts/training_moral_values/train.py -h`

### Training Agreement Analysis models
See `python3 scripts/training_agreement/train.py -h`

## Reproducing paper figures
Below is a (non-exhaustive) list of scripts you need to run to compute the values as presented in the paper.
- **Figure 2**: `scripts/analyze_profiles.py`
- **Table 3**: `scripts/count_debagreements.py` for the most significant value in each subcorpus, and `scripts/analyze_value_conflict.py` for the mean tau distance.
