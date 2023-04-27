#!/bin/bash

## Run your virtual environment activation script
source ~/.virtualenvs/dg_venv3.6/bin/activate
source scripts/train_function.bash

dataset_split="train"
pooling="netvlad"

run_mode="normal"  # Use all GPU and save result to txt files
#run_mode="debug"  # Use a single GPU (to avoid parallel thread in debug mode) and print out result on the screen, and use saved cache features for quickness

#if true ; then
if false ; then
	dataset="pittsburgh"
	run_cluster ${dataset} "clustering" ${run_mode} --arch="alexnet"
fi

if true ; then
#if false ; then
	run_cluster ${dataset} "clustering" ${run_mode} --arch="vgg16"
fi

#if true ; then
if false ; then
	run_cluster ${dataset} "clustering" ${run_mode} --arch="vgg16" --deattention
fi
