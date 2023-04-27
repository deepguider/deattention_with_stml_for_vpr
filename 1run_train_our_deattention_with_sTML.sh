#!/bin/bash

## Run your virtual environment activation script
source ~/.virtualenvs/dg_venv3.6/bin/activate
source scripts/train_function.bash

run_mode="normal"  # Use all GPU and save result to txt files
#run_mode="debug"  # Use a single GPU (to avoid parallel thread in debug mode) and print out result on the screen, and use saved cache features for quickness

nEpochs=30
#nEpochs=10  # for quick train
dataset="pittsburgh"
dataset_split="train"
pooling="netvlad"
clutter_iteration=3  # no affect here

margin=0.1   # default 0.1
tml2_pn_margin=1.4  # default 1.4
deatt_w=0.1

## 1. Train baseline (vgg16+netvlad) + sTML (sharpened triplet margnial loss)
exDesc="baseline_sTML"
set_reference_centroid # Set centroid path
check_centroid
margin=0.1;tml2_pn_margin=1.4  # default 1.4
options="--margin=${margin} --tml2 --tml2_pn_margin=${tml2_pn_margin} --dataloader_margin=1.5"
run_train ${dataset} ${dataset_split} ${pooling} ${exDesc} ${run_mode} ${options}


## 2. Train baseline + deattention
exDesc="baseline_deattention"
set_deattention_centroid # Set centroid path
check_centroid
options="--margin=${margin} --dataloader_margin=${margin} --deatt_padding_mode='reflect'"
run_train_deatt  ${dataset} ${dataset_split} ${pooling} ${deatt_w} ${clutter_iteration} ${exDesc} ${run_mode} ${options}


## 3. Train baseline + deattention + sTML
exDesc="baseline_sTML_deattention"
set_deattention_centroid # Set centroid path
check_centroid
options="--margin=${margin} --tml2 --tml2_pn_margin=${tml2_pn_margin} --dataloader_margin=1.5 --deatt_padding_mode='reflect'"
run_train_deatt  ${dataset} ${dataset_split} ${pooling} ${deatt_w} ${clutter_iteration} ${exDesc} ${run_mode} ${options} --senet_attention --deattention_version=1 --deatt_category_list human vehicle sky
