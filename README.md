# multimodal-timelines

Code for our PAKDD2024 paper: "Using Multimodal Data to Improve Precision of Inpatient Event Timelines". Lecture Notes in Artificial Intelligence, Springer, 2024.  
Authors: Gabriel Frattallone-Llado\*, Juyong Kim\*, Cheng Cheng, Diego Salazar, Smitha Edakalavan, and Jeremy C. Weiss

## Setting up

- Prepare the MIMIC-III dataset (v1.4)
- Setting up the environment:
```
$ conda create --name multimodal-timelines python=3.9
$ conda activate multimodal-timelines
$ conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 r-base=4.2 r-essentials=4.2 -c pytorch -c nvidia -c conda-forge
$ pip install -r requirements.txt
$ python -m spacy download en_core_web_sm
```

## Preprocessing dataset

- Create the initial structured data by running `mimic3buildtimeline_pakdd.R`. Please set the working directory and the MIMIC-III directory before running it.
- Create the absolute timeline dataset and the structured data by running `preprocess_dataset.ipynb`.

## Run code

- Unimodal absolute timeline prediction (3-class classification)
```
# Train/eval on the first fold (5-fold CV) with random seed 42.
$ python main.py --config-name=bert_cls_m3c_cv seed=42 cv_idx=0
$ python main.py --config-name=bert_cls_m3c_cv seed=42 cv_idx=0 test=true

# Train/eval for all 5 folds
$ python main.py --config-name=bert_cls_m3c_cv seed=42 --multirun
$ python main.py --config-name=bert_cls_m3c_cv seed=42 test=true --multirun
```
- Multimodal absolute timeline prediction (3-class classification)
```
# Train/eval on the first fold. It requires 45GB VRAM.
# Multi-GPU training is enabled with num_gpus option. Currently, num_gpus > 1 will utillize all available GPUs.
$ python main.py --config-name=bert_cls_m3c_attn_cv seed=42 cv_idx=0 trainer.params.num_gpus=2
$ python main.py --config-name=bert_cls_m3c_attn_cv seed=42 cv_idx=0 trainer.params.num_gpus=2 test=true

# Train/eval for all 5 folds
$ python main.py --config-name=bert_cls_m3c_attn_cv seed=42 trainer.params.num_gpus=2 --multirun
$ python main.py --config-name=bert_cls_m3c_attn_cv seed=42 trainer.params.num_gpus=2 test=true --multirun
```
