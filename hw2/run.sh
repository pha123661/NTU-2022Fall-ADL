python MultipleChoice.py \
--model_name_or_path="./MC_ckpt/" \
--cache_dir="./cache" \
\
--test_file="${2}" \
--context_file="${1}" \
--preprocessing_num_workers=6 \
--per_device_eval_batch_size=4 \
\
--output_dir="./tmp" \
--do_predict \
--output_file="./inference_test.json"  \
--dataloader_num_workers=6

python QuestionAnswering.py \
--model_name_or_path="./QA_ckpt/" \
--do_predict \
\
--test_file="./inference_test.json" \
--context_file="${1}" \
--output_file="${3}"