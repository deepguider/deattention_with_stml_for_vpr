#!/bin/bash

## Run your virtual environment activation script
source ~/.virtualenvs/dg_venv3.6/bin/activate
source scripts/train_function.bash

run_mode="normal"  # Use all GPU and save result to txt files
#run_mode="debug"  # Use a single GPU (to avoid parallel thread in debug mode) and print out result on the screen, and use saved cache features for quickness

echo "Run train task with varing attention"
nEpochs=30
#nEpochs=10  # for quick train
dataset="pittsburgh"
dataset_split="train"
pooling="netvlad"

## Set centroid path
set_reference_centroid
check_centroid

## Train baseline (vgg16 + netvlad)
run_train ${dataset} ${dataset_split} ${pooling} "baseline" ${run_mode}

## Train baseline with existing attentions
run_train  ${dataset} ${dataset_split} ${pooling} "baseline_crn_attention" ${run_mode} --crn_attention
run_train  ${dataset} ${dataset_split} ${pooling} "baseline_senet_attention" ${run_mode} --senet_attention
run_train  ${dataset} ${dataset_split} ${pooling} "baseline_bam_attention" ${run_mode} --bam_attention
run_train  ${dataset} ${dataset_split} ${pooling} "baseline_cbam_attention" ${run_mode} --cbam_attention
run_train  ${dataset} ${dataset_split} ${pooling} "baseline_ch_attention" ${run_mode} --ch_attention
