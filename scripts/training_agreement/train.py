import json
from pathlib import Path

import torch
from ray import tune
from transformers import (AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from value_disagreement.datasets import DebagreementDataset
from value_disagreement.datasets.utils import cast_dataset_to_hf, hf_dataset_tokenize
from value_disagreement.evaluation import single_label_metrics_cls
from value_disagreement.extraction import (ContextOnlyModel, DebagreementTokenizer,
                                OverriddenBertForSequenceClassification,
                                TransformerUserContextClassifier)
from value_disagreement.utils import print_stats, seed_everything, wandb_get_name


def compute_metrics(p):
    # Compute metrics function
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    return single_label_metrics_cls(preds, p.label_ids)


def model_init(args, id2label, label2id):
    if args.user_context is not None:
        if "user_centroids" in args.user_context:
            try:
                extras_dim = int(Path(args.user_context).stem.split("_")[-1])
            except:
                print("Could not parse extras_dim from user_context, assuming default of 1000")
                extras_dim = 1000
        elif "user_values" in args.user_context:
            # there are 10 values
            extras_dim = 10
        elif "user_features" in args.user_context:
            # there are 9 user features
            extras_dim = 9
        elif "user_noise" in args.user_context:
            extras_dim = 768
        else:
            raise ValueError("Could not estimate number of extra dimensions from user_context")
        if args.only_context:
            return ContextOnlyModel(
                context_dims=extras_dim,
                num_layers=2,
                hidden_dims=300,
                num_labels=len(id2label),
                problem_type="single_label_classification"
            )
        else:
            return TransformerUserContextClassifier.from_pretrained(
                args.use_model,
                extras_dim=extras_dim,
                num_labels=len(label2id),
                mlp_layers=1,
                id2label=id2label,
                label2id=label2id,
                problem_type="single_label_classification"
            )
    else:
        if args.use_model == "override_bert":
            print("Training on overridden BERT (using CLS token instead of pooled output)")
            return OverriddenBertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=len(id2label),
                id2label=id2label,
                label2id=label2id,
                problem_type="single_label_classification"
            )
        else:
            return AutoModelForSequenceClassification.from_pretrained(
                args.use_model,
                num_labels=len(id2label),
                id2label=id2label,
                label2id=label2id,
                problem_type="single_label_classification"
            )


def load_user_context(user_context_path):
    """
    Load user context from a json file.
    """
    with open(user_context_path, "r") as f:
        user2vector = json.load(f)
    return user2vector


def dump_predictions(args, dataset, results, output_file):
    """
    Dump model predictions to a file.
    """
    with open(output_file, 'w') as f:
        result_dump = {
            'args': vars(args),
            'preds': results.predictions.tolist(),
            'true': dataset['labels'].tolist(),
            'metrics': results.metrics,
        }
        json.dump(result_dump, f)


def main(args):
    # Set seed for reproducibility
    seed_everything(args.seed)

    # Load dataset
    dataset = DebagreementDataset("data/debagreement.csv", sample_N=-1)
    if args.user_context is not None:
        user2vector = load_user_context(args.user_context)
        dataset.filter_authors(user2vector.keys())
        dataset.compute_profile_similarities(user2vector)
    else:
        user2vector = None

    # Split data
    if args.data_split == "random":
        train_len = int(len(dataset) * 0.8)
        val_len = int((len(dataset) - train_len) / 2)
        test_len = len(dataset) - (train_len + val_len)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
    elif args.data_split == "temporal":
        train_set, val_set, test_set = dataset.get_ordered_splits()

    print(f"Train size: {len(train_set)} - val_size: {len(val_set)} - test_size: {len(test_set)}")

    if args.only_subsample:
        user2vector = None
        args.user_context = None

    # Convert pytorch dataset to huggingface dataset
    train_dataset = cast_dataset_to_hf(train_set, "train", dataset_type="agreement", load_context=user2vector, context_append_mode='mixin_token')
    val_dataset = cast_dataset_to_hf(val_set, "val", dataset_type="agreement", load_context=user2vector, context_append_mode='mixin_token')
    test_dataset = cast_dataset_to_hf(test_set, "test", dataset_type="agreement", load_context=user2vector, context_append_mode='mixin_token')

    # Load tokenizer
    agreement_tokenizer = DebagreementTokenizer(args.use_model, args.context_append_mode)
    tokenized_dataset_train = hf_dataset_tokenize(train_dataset, agreement_tokenizer, args.use_model, context=user2vector, context_append_mode=args.context_append_mode)
    tokenized_dataset_val = hf_dataset_tokenize(val_dataset, agreement_tokenizer, args.use_model, context=user2vector, context_append_mode=args.context_append_mode)
    tokenized_dataset_test = hf_dataset_tokenize(test_dataset, agreement_tokenizer, args.use_model, context=user2vector, context_append_mode=args.context_append_mode)

    if args.context_append_mode == "mixin_token":
        args.user_context = None


    # Training arguments
    training_args = TrainingArguments(
        "debagreement_trainer",
        report_to="wandb",
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=150,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        logging_strategy='steps',
        logging_steps=150,
        save_strategy="steps",
        save_steps=150,
        save_total_limit=1,
    )

    data_collator = DataCollatorWithPadding(tokenizer=agreement_tokenizer.tokenizer)
    if args.ray:
        hyper_space = lambda: {
            "learning_rate": tune.loguniform(1e-7, 1e-2),
            "weight_decay": tune.uniform(1e-7, 1e-2),
            "lr_scheduler_type": tune.choice(["cosine", "linear"]),
            "max_grad_norm": tune.uniform(0.1, 1.0),
            "warmup_ratio": tune.uniform(0.01, 0.3),
            "num_train_epochs": tune.choice(list(range(15))),
        }
        trainer = Trainer(
            args=training_args,
            model_init=lambda: model_init(args, dataset.id2label, dataset.label2id),
            train_dataset=tokenized_dataset_train,
            eval_dataset=tokenized_dataset_val,
            data_collator=data_collator,
            tokenizer=agreement_tokenizer.tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.hyperparameter_search(
            hp_space=hyper_space,
            compute_objective=lambda x: x['eval_f1'],
            direction="maximize",
            backend="ray",
            n_trials=20,
            local_dir="~/projects/implicit/aspects/ray_results/"
        )
    else:
        reses = []
        for i in range(args.n_runs):
            seed_everything(i)
            model = model_init(args, dataset.id2label, dataset.label2id)
            run_name = wandb_get_name()
            training_args.run_name = run_name
            trainer = Trainer(
                model,
                training_args,
                train_dataset=tokenized_dataset_train,
                eval_dataset=tokenized_dataset_val,
                data_collator=data_collator,
                tokenizer=agreement_tokenizer.tokenizer,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            if args.user_context is not None:
                model_name = f"agreement_trainer_{args.use_model}_{args.user_context.replace('data/', '').replace('.json', '')}_{i}"
            else:
                model_name = f"agreement_trainer_{args.use_model}_nocontext_{i}"
            results = trainer.predict(tokenized_dataset_test)
            dump_predictions(args, tokenized_dataset_test, results, f"output/predictions/{model_name}.json")
            trainer.save_model(model_name)
            reses.append({
                'model_name': model_name,
                'results': results,
            })
        print_stats("Precision", [res['results'].metrics['test_precision'] for res in reses])
        print_stats("Recall", [res['results'].metrics['test_recall'] for res in reses])
        print_stats("F1", [res['results'].metrics['test_f1'] for res in reses])
        print_stats("Accuracy", [res['results'].metrics['test_accuracy'] for res in reses])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_model', type=str, default='bert-base-uncased', help="model from hf hub to use")
    parser.add_argument('--data_split', type=str, default='temporal', choices=['random', 'temporal'],
                        help="how to split the debagreement data into sets")
    parser.add_argument('--user_context', type=str, default=None,
                        help="If defined, use this path for loading user context vectors")
    parser.add_argument('--only_context', default=False, action='store_true',
                        help="If true, only use user context vectors for predicting agreement")
    parser.add_argument('--only_subsample', action='store_true',
                        help="If True, only subsample the data (user context vectors are provided, but not used in the model)")
    parser.add_argument('--seed', type=int, default=0,
                        help="Experiment seed")
    parser.add_argument('--n_runs', type=int, default=1,
                        help="How many times to train and evaluate the model")
    parser.add_argument('--num_epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=5e-05,
                        help="Learning rate for training")
    parser.add_argument('--ray', default=False, action='store_true',
                        help="If true, use ray for distributed hyperparameter search.")
    parser.add_argument('--context_append_mode', default="concat_vector", choices=["concat_vector", "mixin_token"],
                        help="Mode to use for concatenating context vectors to the BERT embeddings")
    args = parser.parse_args()
    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(args)
