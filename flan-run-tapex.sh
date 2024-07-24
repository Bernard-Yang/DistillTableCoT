Gen=flan-t5-small-cot.txt
# split=test

CUDA_VISIBLE_DEVICES=0 python eval_contlog_with_tapex.py \
--do_predict \
--model_name_or_path microsoft/tapex-large-finetuned-tabfact \
--test_name ../model_outputs/scigen/$Gen \
--split_name test \
--output_dir ./tapex-contlog-eval \
--affix flan-t5-small \
--data_dir ../data/scigen \
--per_device_eval_batch_size 24 \
--eval_accumulation_steps 6       

CUDA_VISIBLE_DEVICES=0 python3 eval_contlog_with_tapas.py \
--data_dir ../data/scigen \
--batch_size 1 \
--split_name test \
--test_file ../model_outputs/scigen/$Gen
