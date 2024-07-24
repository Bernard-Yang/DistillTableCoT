
DATA_DIRECTORY=./cot
MODEL_NAME=google/flan-t5-small 
BATCH_SIZE=32
OUTPUT_DIR=./flan-t5-small-train/
ckpt="./flan-t5-small/val_mover=0.00-step_count=6.ckpt"
# cp -R $OUTPUT_DIR ./ckpt/flan-t5-base-Large
rm -rf $OUTPUT_DIR
mkdir $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=1 python train_table2text_flan_t5.py \
--data_dir=$DATA_DIRECTORY \
--model_name_or_path=$MODEL_NAME \
--learning_rate=5e-5 \
--num_train_epochs 15 \
--train_batch_size=$BATCH_SIZE \
--eval_batch_size=$BATCH_SIZE \
--test_batch_size=$BATCH_SIZE \
--output_dir=$OUTPUT_DIR \
--n_gpu 1 \
--do_train \
--do_predict \
--early_stopping_patience 10 \
--max_source_length 384 \
--max_target_length 384 \