python MultipleChoice.py \
--model_name_or_path="./MC_ckpt" \
--cache_dir="./cache" \
\
--test_file="./data/test.json" \
--context_file="./data/context.json" \
--preprocessing_num_workers=6 \
\
--output_dir="./tmp" \
--do_predict \
--output_file="./inference_test.json"  \
--dataloader_num_workers=6 \