python jsonl2json.py --input="${1}" --output="test_input.json"

python Inference_sum.py \
--model_name_or_path="./RL_ckpt" \
--num_beams=5 \
\
--test_file="test_input.json" \
--text_column="maintext" \
--preprocessing_num_workers=6 \
\
--output_dir="./" \
--overwrite_output_dir=true \
--dataloader_num_workers=4 \
\
--predict_with_generate=True \

python process_gen_rst.py --test_jsonl="${1}" --output="${2}"