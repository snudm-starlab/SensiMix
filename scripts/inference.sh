# Make fast inference
# Fix the file path according to your environment

export GLUE_DIR=/home/piaotairen/data/glue_data
export TASK_NAME=SST-2

CUDA_VISIBLE_DEVICES="0" python ../src/examples/run_glue_inference.py \
  --model_type mpqbert \
  --model_name_or_path /home/piaotairen/experiments/test_output/$TASK_NAME/quantized \
  --tokenizer_name bert-base-uncased \
  --config_name /home/piaotairen/SensiMix/src/config.json \
  --task_name $TASK_NAME \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size 128 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 2.5e-5 \
  --num_train_epoch 1 \
  --output_dir /home/piaotairen/experiments/test_output/$TASK_NAME/quantized \
  --quantized_model_dir /home/piaotairen/experiments/test_output/$TASK_NAME/quantized \
  --overwrite_output_dir \
