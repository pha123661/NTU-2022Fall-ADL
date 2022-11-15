python Inference_sum.py \
--model_name_or_path="./Sum_ckpt" \
--num_beams=3 \
\
--test_file="./data/sample_test.json" \
--text_column="maintext" \
--preprocessing_num_workers=6 \
\
--output_dir="./" \
--overwrite_output_dir=true \
--dataloader_num_workers=4 \
\
--predict_with_generate=True \