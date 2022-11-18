python RL_Summarization.py \
--model_name_or_path="./Sum_ckpt" \
--cache_dir="./cache" \
--num_beams=5 \
--generation_max_length=20 \
\
--train_file="./data/train.json" \
--validation_file="./data/public.json" \
--text_column="maintext" \
--summary_column="title" \
--preprocessing_num_workers=6 \
\
--output_dir="./RL_fine_tune_ckpt" \
--do_train \
--do_eval \
--num_train_epochs=10 \
--auto_find_batch_size \
--gradient_accumulation_steps=4 \
--learning_rate=5e-5 \
--warmup_steps=300 \
--lr_scheduler_type=cosine \
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
