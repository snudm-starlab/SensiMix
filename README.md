# SensiMix
This repository provides implementations of SensiMix: Sensitivity-Aware 8-bit Index & 1-bit Value Mixed Precision Quantization for BERT Compression.

## Overview
#### Code structure
```
SensiMix
  │ 
  ├── src
  │    │     
  │    ├── models
  │    │     ├── configuration_auto.py: Config class
  │    │     ├── configuration_mpqbert: MPQBERT's config
  │    │     ├── modeling_mpqbert.py: Model of MPQBERT
  │    │     ├── modeling_mpqbert_infer.py: Make fast inference
  │    │     ├── optimization.py: Optimization for BERT model
  │    │     └── quantized_modules.py: Quantized operations
  │    │      
  │    └── examples
  │          ├── run_glue_new.py: train/validate on glue dataset 
  |          └── download_glue_data.py: script for dowloading the GLUE benchmark tasks
  │    
  │    
  │
  └── script: shell scripts for running the training/testing codes
```


#### Data description
* GLUE benchmark dataset [[Homepage]](https://gluebenchmark.com/)
    * Visit the official hompage to check the detail information.
   
#### Output
* The trained full-precision (32-bit) model
* The quantized SensiMix model 

## Installation
#### Environment 
* Unbuntu 16.04 (LTS)
* CUDA 10.1
* Python 3.7
* PyTorch 1.7.1 (CUDA 10.1)
* torchvision
* transformers 2.5.1

Notice: Please install the PyTorch 1.7.1 with CUDA 10.1 extension


## How to use 
#### Clone the repository
    git clone https://github.com/snudm-starlab/SensiMix.git
    cd SensiMix

#### Install the dependencies
    pip install -r requirements.txt

#### Setup the xnor extension
    cd src/xnor/cuda
    python setup.py install

#### Run the demo.
    make

#### Training & Evaluating
* Fine tuning the SensiMix model on the GLUE benchmark datasets, run script:
    ```    
    cd script
    bash ft.sh
    ```
    * All settings are in the run_glue_new.py
    * You can write your own script.
* Full-precision SensiMix model and quantized SensiMix model will be saved in the directory you set in the script ft.sh.

* For example, your training script should like
    ```
    export GLUE_DIR=/data/glue_data
    export TASK_NAME=MRPC

    CUDA_VISIBLE_DEVICES="0" python ../src/examples/run_glue_new.py \
        --model_type mpqbert \
        --model_name_or_path bert-base-uncased \
        --tokenizer_name bert-base-uncased \
        --config_name /home/piaotairen/SensiMix/src/config.json \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --evaluate_during_training \
        --logging_steps 200 \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --save_steps 100000 \
        --per_gpu_eval_batch_size 128 \
        --per_gpu_train_batch_size 16 \
        --learning_rate 3.0e-5 \
        --num_train_epoch 6 \
        --output_dir ./experiments/$TASK_NAME/fp_model/ \
        --save_quantized_model \
        --overwrite_cache \
        --overwrite_output_dir \
        --quantized_model_dir ./experiments/$TASK_NAME/quantized/ \
    ```

#### Fast inference
* Run run_glue_inference.py to make fast inference
    * Load the SensiMix model that was saved.
    * Run the inference.sh to make fast inference.

* For example, your inference script should like
    ```
    export GLUE_DIR=/home/piaotairen/data/glue_data_inference
    export TASK_NAME=QNLI

    CUDA_VISIBLE_DEVICES="0" python ../src/examples/run_glue_inference.py \
        --model_type mpqbert \
        --model_name_or_path ./experiments/fine_tuning/$TASK_NAME/quantized \
        --tokenizer_name bert-base-uncased \
        --config_name /home/piaotairen/SensiMix/src/config.json \
        --task_name $TASK_NAME \
        --do_eval \
        --logging_steps 1000000 \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size 128 \
        --per_gpu_train_batch_size 16 \
        --learning_rate 2.5e-5 \
        --num_train_epoch 1 \
        --output_dir ./experiments/$TASK_NAME/quantized \
        --quantized_model_dir ./experiments/$TASK_NAME/quantized \
        --overwrite_output_dir \
    ```

## Contact us
- Tairen Piao (piaotairen@snu.ac.kr)
- Ikhyun Cho (ikhyuncho@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

*This software may be used only for research evaluation purposes.*
*For other purposes (e.g., commercial), please contact the authors.*
