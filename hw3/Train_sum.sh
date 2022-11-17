python Summarization.py \
--model_name_or_path="google/mt5-small" \
--cache_dir="./cache" \
--num_beams=3 \
\
--train_file="./data/train.json" \
--validation_file="./data/public.json" \
--text_column="maintext" \
--summary_column="title" \
--preprocessing_num_workers=6 \
\
--output_dir="./Sum_ckpt" \
--do_train \
--do_eval \
--num_train_epochs=100 \
--auto_find_batch_size \
--gradient_accumulation_steps=4 \
--learning_rate=5e-5 \
--warmup_steps=300 \
--dataloader_num_workers=4 \
\
--evaluation_strategy="steps" \
--eval_steps=500 \
--save_steps=500 \
--metric_for_best_model="rouge_combined" \
--load_best_model_at_end  \
--report_to="tensorboard" \
--predict_with_generate=True \
\
--bf16 --tf32=y \