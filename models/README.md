# Baseline Models

For VRDP and NS-DR, since we do not perform any finetuning or training-from-scratch, we directly use the officially released code at:

- VRDP: https://github.com/dingmyu/VRDP
- NS-DR: https://github.com/chuangg/CLEVRER


### ALOE Variants

For all ALOE-related variants, you should first follow the original README (inside the `object_attention_for_reasoning` folder) to download the pretrained MONet Latents.

After unzipping, they will be put inside a folder named `clevrer_monet_latents`. Next, you need to copy-paste (or symbol-link) all files inside folder `object_attention_for_reasoning_data`
into the `clevrer_monet_latents` folder. It contains processed data of the original CLEVRER-Humans into ALOE-compatible formats.

- Training from scratch: enter `object_attention_for_reasoning` and run `CUDA_VISIBLE_DEVICES=0 python3 train_model.py`.
- Pretraining and zero-shot testing: enter `object_attention_for_reasoning_fixed` and run `CUDA_VISIBLE_DEVICES=0 python3 run_model.py`.
- Pretraining and finetuning: enter `object_attention_for_reasoning_finetune` and run `CUDA_VISIBLE_DEVICES=0 python3 train_model.py`.


### CNN+LSTM and CNN+BERT

Download: 
- Annotations, and CEGs from our [website](https://sites.google.com/stanford.edu/clevrer-humans/home). 
- Cache of pre-computed CNN features for all videos from the [drive](https://drive.google.com/drive/folders/1FQ4AsQnjjCqXe9_oschZ8xQebr5tcz8x?usp=sharing).

To train the models on CLEVRER-Humans from scratch:

`python train.py --model_type MODEL_TYPE--output_dir OUTPUT_DIR --graph_path CEG_PATH  --use_raw_qa True --train_phase cegv3_core_train --validation_phase cegv3_core_val`

To pretrain the models on CLEVRER:

`python3 train.py --model_type MODEL_TYPE --output_dir OUTPUT_DIR --graph_path CEG_PATH  --use_raw_qa True --train_phase train --validation_phase validation`

To finetune the pretrained models on CLEVRER-Humans:

`python3 train.py --model_type MODEL_TYPE--output_dir OUTPUT_DIR --graph_path CEG_PATH  --use_raw_qa True --model_name_or_path OUTPUT_DIR_PRETRAINED  --train_phase cegv3_core_val --validation_phase cegv3_core_val`

`MODEL_TYPE: resnet50_lstm / resnet50_bert`

