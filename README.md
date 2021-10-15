# SensiMix
This repository provides the implementation of SensiMix: Sensitivity-Aware 8-bit Index & 1-bit Value Mixed Precision Quantization for BERT Compression.

## Overview
#### Code structure
```
SensiMix
  │ 
  ├── src
  │    │     
  │    ├── models
  │    │     ├── configuration_auto.py: configuration class
  │    │     ├── configuration_mpqbert.py: configuration of the SensiMix model
  │    │     ├── modeling_mpqbert.py: the SensiMix model
  │    │     ├── modeling_mpqbert_infer.py: the SensiMix inference model 
  │    │     ├── optimization.py: optimization classes for the BERT model
  │    │     └── quantized_modules.py: quantized classes and functions
  │    │      
  │    └── examples
  │          ├── run_glue_new.py: train/validate the models on the GLUE benchmark tasks 
  |          └── run_glue_inference.py: load the quantized model and make inference only
  │    
  │    
  └── scripts: shell scripts for training and testing
```


#### Datasets description
* GLUE benchmark tasks [[Homepage]](https://gluebenchmark.com/)
    * Visit the official hompage to check the detail information.
    * You can download the datasets on the website.
   

## Dependencies
* Ubuntu 16.04 (LTS)
* Python 3.7
* PyTorch 1.7.1 (CUDA 10.1)
* transformers 2.5.1

Notice: Please install the PyTorch 1.7.1 with CUDA 10.1 extension


## How to use 
#### Clone the repository
    git clone https://github.com/snudm-starlab/SensiMix.git
    cd SensiMix

#### Install the required packages
    pip install -r requirements.txt
    
If other packages are required, use "pip install" to install them.

#### Install the xnor extension
    cd src/xnor/cuda
    python setup.py install

#### Run the demo
    bash demo.sh

#### Training & Evaluation
* Fine-tune the SensiMix model on the GLUE benchmark datasets, run script:
    ```    
    cd scripts
    bash ft.sh
    ```
* All the argument settings are in the run_glue_new.py


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
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size 128 \
        --per_gpu_train_batch_size 16 \
        --learning_rate 3.0e-5 \
        --num_train_epoch 6 \
        --output_dir ./experiments/$TASK_NAME/fp_model/ \
        --save_quantized_model \
        --overwrite_output_dir \
        --quantized_model_dir ./experiments/$TASK_NAME/quantized/ \
    ```
* Output
    * The trained full-precision (32-bit) model.
    * The quantized SensiMix model.

#### Fast inference
* Run inference.sh to make fast inference, run script:
    ```    
    cd scripts
    bash inference.sh
    ```

* For example, your inference script should like
    ```
    export GLUE_DIR=/data/glue_data
    export TASK_NAME=MRPC

    CUDA_VISIBLE_DEVICES="0" python ../src/examples/run_glue_inference.py \
        --model_type mpqbert \
        --model_name_or_path ./experiments/$TASK_NAME/quantized \
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
        --output_dir ./experiments/$TASK_NAME/quantized \
        --quantized_model_dir ./experiments/$TASK_NAME/quantized \
        --overwrite_output_dir \
    ```
    
## Contact us
- Tairen Piao ( piaotairen@snu.ac.kr)
- Ikhyun Cho ( ikhyuncho@snu.ac.kr)
- U Kang ( ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

*This software may be used only for research evaluation purposes.*
*For other purposes (e.g., commercial), please contact the authors.*
