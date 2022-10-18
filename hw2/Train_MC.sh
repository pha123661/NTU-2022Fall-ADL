# https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/trainer#transformers.TrainingArguments
python MultipleChoice.py  \
--model_name_or_path="hfl/chinese-macbert-large" \
--cache_dir="./cache" \
\
--train_file="./data/train.json" \
--validation_file="./data/valid.json" \
--context_file="./data/context.json" \
--preprocessing_num_workers=6 \
\
--output_dir="./MC_ckpt" \
--do_train \
--do_eval \
--num_train_epochs=9 \
--auto_find_batch_size \
--learning_rate=2e-5 \
--warmup_steps=300 \
--bf16 --tf32=y \
--dataloader_num_workers=6 \
\
--evaluation_strategy="steps" \
--eval_steps=500 \
--save_steps=500 \
--metric_for_best_model="accuracy" \
--load_best_model_at_end  \
--report_to="tensorboard" \