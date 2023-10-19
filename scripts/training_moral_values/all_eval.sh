#!/bin/bash

# BERT
> scripts/training_moral_values/results_bert_va_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valueeval_0/ --eval_only >> scripts/training_moral_values/results_bert_va_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valueeval_1/ --eval_only >> scripts/training_moral_values/results_bert_va_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valueeval_2/ --eval_only >> scripts/training_moral_values/results_bert_va_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valueeval_3/ --eval_only >> scripts/training_moral_values/results_bert_va_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valueeval_4/ --eval_only >> scripts/training_moral_values/results_bert_va_vn.txt

> scripts/training_moral_values/results_bert_vn_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valuenet_0/ --eval_only >> scripts/training_moral_values/results_bert_vn_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valuenet_1/ --eval_only >> scripts/training_moral_values/results_bert_vn_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valuenet_2/ --eval_only >> scripts/training_moral_values/results_bert_vn_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valuenet_3/ --eval_only >> scripts/training_moral_values/results_bert_vn_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valuenet_4/ --eval_only >> scripts/training_moral_values/results_bert_vn_va.txt

> scripts/training_moral_values/results_bert_va_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valueeval_0/ --eval_only >> scripts/training_moral_values/results_bert_va_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valueeval_1/ --eval_only >> scripts/training_moral_values/results_bert_va_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valueeval_2/ --eval_only >> scripts/training_moral_values/results_bert_va_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valueeval_3/ --eval_only >> scripts/training_moral_values/results_bert_va_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valueeval_4/ --eval_only >> scripts/training_moral_values/results_bert_va_both.txt

> scripts/training_moral_values/results_bert_vn_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valuenet_0/ --eval_only >> scripts/training_moral_values/results_bert_vn_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valuenet_1/ --eval_only >> scripts/training_moral_values/results_bert_vn_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valuenet_2/ --eval_only >> scripts/training_moral_values/results_bert_vn_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valuenet_3/ --eval_only >> scripts/training_moral_values/results_bert_vn_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_valuenet_4/ --eval_only >> scripts/training_moral_values/results_bert_vn_both.txt

> scripts/training_moral_values/results_bert_both_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_both_0/ --eval_only >> scripts/training_moral_values/results_bert_both_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_both_1/ --eval_only >> scripts/training_moral_values/results_bert_both_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_both_2/ --eval_only >> scripts/training_moral_values/results_bert_both_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_both_3/ --eval_only >> scripts/training_moral_values/results_bert_both_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_both_4/ --eval_only >> scripts/training_moral_values/results_bert_both_vn.txt

> scripts/training_moral_values/results_bert_both_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_both_0/ --eval_only >> scripts/training_moral_values/results_bert_both_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_both_1/ --eval_only >> scripts/training_moral_values/results_bert_both_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_both_2/ --eval_only >> scripts/training_moral_values/results_bert_both_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_both_3/ --eval_only >> scripts/training_moral_values/results_bert_both_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model bert-base-uncased --checkpoint moral_values_trainer_bert-base-uncased_both_4/ --eval_only >> scripts/training_moral_values/results_bert_both_va.txt


# RoBERTa
> scripts/training_moral_values/results_roberta_both_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_both_0/ --eval_only >> scripts/training_moral_values/results_roberta_both_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_both_1/ --eval_only >> scripts/training_moral_values/results_roberta_both_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_both_2/ --eval_only >> scripts/training_moral_values/results_roberta_both_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_both_3/ --eval_only >> scripts/training_moral_values/results_roberta_both_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_both_4/ --eval_only >> scripts/training_moral_values/results_roberta_both_vn.txt

> scripts/training_moral_values/results_roberta_both_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_both_0/ --eval_only >> scripts/training_moral_values/results_roberta_both_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_both_1/ --eval_only >> scripts/training_moral_values/results_roberta_both_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_both_2/ --eval_only >> scripts/training_moral_values/results_roberta_both_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_both_3/ --eval_only >> scripts/training_moral_values/results_roberta_both_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_both_4/ --eval_only >> scripts/training_moral_values/results_roberta_both_va.txt

> scripts/training_moral_values/results_roberta_vn_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valuenet_0/ --eval_only >> scripts/training_moral_values/results_roberta_vn_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valuenet_1/ --eval_only >> scripts/training_moral_values/results_roberta_vn_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valuenet_2/ --eval_only >> scripts/training_moral_values/results_roberta_vn_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valuenet_3/ --eval_only >> scripts/training_moral_values/results_roberta_vn_va.txt
python3 scripts/training_moral_values/train_paper.py valueeval --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valuenet_4/ --eval_only >> scripts/training_moral_values/results_roberta_vn_va.txt

> scripts/training_moral_values/results_roberta_va_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valueeval_0/ --eval_only >> scripts/training_moral_values/results_roberta_va_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valueeval_1/ --eval_only >> scripts/training_moral_values/results_roberta_va_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valueeval_2/ --eval_only >> scripts/training_moral_values/results_roberta_va_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valueeval_3/ --eval_only >> scripts/training_moral_values/results_roberta_va_vn.txt
python3 scripts/training_moral_values/train_paper.py valuenet --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valueeval_4/ --eval_only >> scripts/training_moral_values/results_roberta_va_vn.txt

> scripts/training_moral_values/results_roberta_va_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valueeval_0/ --eval_only >> scripts/training_moral_values/results_roberta_va_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valueeval_1/ --eval_only >> scripts/training_moral_values/results_roberta_va_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valueeval_2/ --eval_only >> scripts/training_moral_values/results_roberta_va_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valueeval_3/ --eval_only >> scripts/training_moral_values/results_roberta_va_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valueeval_4/ --eval_only >> scripts/training_moral_values/results_roberta_va_both.txt

> scripts/training_moral_values/results_roberta_vn_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valuenet_0/ --eval_only >> scripts/training_moral_values/results_roberta_vn_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valuenet_1/ --eval_only >> scripts/training_moral_values/results_roberta_vn_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valuenet_2/ --eval_only >> scripts/training_moral_values/results_roberta_vn_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valuenet_3/ --eval_only >> scripts/training_moral_values/results_roberta_vn_both.txt
python3 scripts/training_moral_values/train_paper.py both --use_model roberta-base --checkpoint moral_values_trainer_roberta-base_valuenet_4/ --eval_only >> scripts/training_moral_values/results_roberta_vn_both.txt