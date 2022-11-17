python RL_finetune.py \
--model_name_or_path="./Sum_ckpt" \
--cache_dir="./cache" \
--num_beams=3 \
\
--train_file="./data/train.json" \
--validation_file="./data/public.json" \
--text_column="maintext" \
--summary_column="title" \
--preprocessing_num_workers=6 \
--output_dir="./RL_ckpt" \
--predict_with_generate=True \
--report_to="tensorboard" \