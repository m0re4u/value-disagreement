import copy

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BertForSequenceClassification, BertModel,
                          PreTrainedModel, Trainer, TrainingArguments)

from value_disagreement.datasets import (RedditAnnotatedDataset, ValueEvalDataset,
                              ValueNetDataset)
from value_disagreement.datasets.utils import cast_dataset_to_hf, hf_dataset_tokenize
from value_disagreement.extraction import ValueConstants, ValueTokenizer


def model_init(checkpoint, tokenizer):
    """
    Initialize the model with the checkpoint and tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=1,
    )
    model.resize_token_embeddings(len(tokenizer.tokenizer))
    return model


proj_dir = "./"
MODEL_NAME = "bert-base-uncased"
dataset_net = ValueNetDataset(
    f"{proj_dir}/data/valuenet/",
    return_predefined_splits=True
)
print(dataset_net[3])

train_vn_idx, val_vn_idx, test_vn_idx = dataset_net.get_splits()
train_hf = cast_dataset_to_hf(dataset_net[train_vn_idx], "train", abs_label=False)
val_hf = cast_dataset_to_hf(dataset_net[val_vn_idx], "val", abs_label=False)
test_hf = cast_dataset_to_hf(dataset_net[test_vn_idx], "test", abs_label=False)

value_tokenizer = ValueTokenizer(
    MODEL_NAME,
    input_concat=True,
    label_type="cast_float"
)

print(test_hf[3])

# Tokenize data
tokenized_dataset_train = hf_dataset_tokenize(train_hf, value_tokenizer, MODEL_NAME, soft_target_type='float')
tokenized_dataset_val = hf_dataset_tokenize(val_hf, value_tokenizer, MODEL_NAME, soft_target_type='float')
tokenized_dataset_test = hf_dataset_tokenize(test_hf, value_tokenizer, MODEL_NAME, soft_target_type='float')

print(tokenized_dataset_test[3])
# Add special value tokens and initalize model
num_added_tokens = value_tokenizer.tokenizer.add_special_tokens(
    {"additional_special_tokens": [f"<{x}>" for x in ValueConstants.SCHWARTZ_VALUES]})
print(f"Added {num_added_tokens} special value tokens")
model = model_init("checkpoint-495/", value_tokenizer)

batch_size = 256

# metric = evaluate.load("mse")

# training_args = TrainingArguments(
#     output_dir="qiu_trainer",
#     learning_rate=5e-06,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=40,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     report_to="wandb",
#     metric_for_best_model="mse",
#     logging_strategy='steps',
#     logging_steps=20,
#     load_best_model_at_end=True,
#     weight_decay=0.00,
# )

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     rmse = metric.compute(predictions=predictions, references=labels, squared=False)
#     return rmse

# trainer = Trainer(
#     model,
#     training_args,
#     train_dataset=tokenized_dataset_train,
#     eval_dataset=tokenized_dataset_val,
#     tokenizer=value_tokenizer.tokenizer,
#     compute_metrics=compute_metrics,
# )
# trainer.train()

def evaluate_model(checkpoint, dataset):
    model = model_init(checkpoint, value_tokenizer)
    args = TrainingArguments(
        output_dir=".",
        do_train=False,
        do_eval=False,
        do_predict=True,
        per_device_eval_batch_size=2
    )

    trainer = Trainer(
        model,
        args,
        tokenizer=value_tokenizer.tokenizer,
    )

    predictions = trainer.predict(dataset)
    pred_int = np.round(predictions.predictions, 0)

    f1_simple_rounding = f1_score(y_true=np.abs(predictions.label_ids), y_pred=np.abs(pred_int), average='macro')
    print(f"F1 score simple rounding: {f1_simple_rounding:.3f}")

evaluate_model("checkpoint-495/", tokenized_dataset_test)

dataset_eval = ValueEvalDataset(
    f"{proj_dir}/data/valueeval/dataset-identifying-the-human-values-behind-arguments/",
    cast_to_valuenet=True,
    return_predefined_splits=True
)
test_set_idx = []
for i, elem in enumerate(dataset_eval):
    if elem['split'] == 'test':
        test_set_idx.append(i)

test_set = dataset_eval[test_set_idx]
test_hf = cast_dataset_to_hf(test_set, "test", abs_label=False)

value_tokenizer = ValueTokenizer(
    MODEL_NAME,
    input_concat=True,
    label_type="cast_float"
)
tokenized_dataset_test_eval = hf_dataset_tokenize(test_hf, value_tokenizer, MODEL_NAME, soft_target_type='float')
# Add special value tokens and initalize model
num_added_tokens = value_tokenizer.tokenizer.add_special_tokens(
    {"additional_special_tokens": [f"<{x}>" for x in ValueConstants.SCHWARTZ_VALUES]})
print(f"Added {num_added_tokens} special value tokens")

evaluate_model("checkpoint-495/", tokenized_dataset_test_eval)

from datasets import concatenate_datasets

big_ds = concatenate_datasets([tokenized_dataset_test, tokenized_dataset_test_eval])
evaluate_model("checkpoint-495/", big_ds)