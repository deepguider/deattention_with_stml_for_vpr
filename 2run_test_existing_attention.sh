#!/bin/bash

## Run your virtual environment activation script
source ~/.virtualenvs/dg_venv3.6/bin/activate
source scripts/test_function.bash

function run_baseline(){  # Run reference's baseline
    # Baseline is from the reference paper : Arandjelovic, Relja, et al. "NetVLAD: CNN architecture for weakly supervised place recognition." CVPR. 2016. 
	# Baseline weight from https://github.com/Nanne/pytorch-NetVlad, which is pytorch implementation of reference paper
    RESUME="pretrained/vgg16_netvlad_checkpoint";ckpt="best";exDesc="referpaper_baseline" 
    run_test ${dataset} ${dataset_split} ${pooling} ${exDesc} ${run_mode} 
    #run_test_add_clutter_in_q_of_test ${dataset} ${dataset_split} ${pooling} ${clutter_iteration} ${exDesc} ${run_mode} 
    #run_test_add_clutter_in_dbq_of_test ${dataset} ${dataset_split} ${pooling} ${clutter_iteration} ${exDesc} ${run_mode} 
}

function run_test_pooling_existing_attention(){  # Run baseline except deattention
	## input parameters
    pooling=$1  # "netvlad", "max", "avg", "gem"
	attention=$2  # --senet_attention --bam_attention --cbam_attention --ch_eca_attention --ch_eca_attention_k_size=3
	trained_dataset=$3  # "pitts30k", "tokyoTM"
	option1=$4
	option2=$5
	## RESUME ex) "pretrained/trained_pitts30k_vgg16_maxpooling_deatt_w0.001_add_clutter3_in_train"
	attention_opt="--${attention}"

	## Our baseline
    RESUME="pretrained/trained_${trained_dataset}_vgg16_${pooling}pooling_attention_${attention}";
	exDesc="trained_with_normal_datasaet_other_attention_${attention}"
    run_test ${dataset} ${dataset_split} ${pooling} ${exDesc} ${run_mode} ${attention_opt} ${option1} ${option2}
    #run_test_add_clutter_in_q_of_test ${dataset} ${dataset_split} ${pooling} ${clutter_iteration} ${exDesc} ${run_mode} ${attention_opt} ${option1} ${option2}
    #run_test_add_clutter_in_dbq_of_test ${dataset} ${dataset_split} ${pooling} ${clutter_iteration} ${exDesc} ${run_mode} ${attention_opt} ${option1} ${option2}
}

## Configuration
run_mode="normal"  # Use all GPU and save result to txt files
#run_mode="debug"  # Use a single GPU (to avoid parallel thread in debug mode) and print out result on the screen, and use saved cache features for quickness

clutter_iteration=3

#if false ; then
if true ; then
	run_baseline
	ckpt="best"
	pooling="netvlad";
	dataset="pittsburgh"; dataset_split="val"; run_baseline ${pooling}
	dataset="pittsburgh"; dataset_split="test";run_baseline ${pooling}
	dataset="tokyo247"; dataset_split="test";run_baseline ${pooling}
	dataset="tokyoTM"; dataset_split="test";run_baseline ${pooling}
fi

#if false ; then
if true ; then
	ckpt="best"
	pooling="netvlad";
	attention="crn_attention"
	trained_dataset="pitts30k"  # These are used to specify RESUME weight file
	dataset="pittsburgh"; dataset_split="val"; run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
	dataset="pittsburgh"; dataset_split="test";run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
	dataset="tokyo247"; dataset_split="test";run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
	dataset="tokyoTM"; dataset_split="test";run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
fi

#if false ; then
if true ; then
	ckpt="best"
	pooling="netvlad";
	attention="senet_attention"
	trained_dataset="pitts30k"  # These are used to specify RESUME weight file
	dataset="pittsburgh"; dataset_split="val"; run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
	dataset="pittsburgh"; dataset_split="test";run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
	dataset="tokyo247"; dataset_split="test";run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
	dataset="tokyoTM"; dataset_split="test";run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
fi

if true ; then
#if false ; then
	ckpt="best"
	pooling="netvlad";
	attention="bam_attention"
	trained_dataset="pitts30k"  # These are used to specify RESUME weight file
	dataset="pittsburgh"; dataset_split="val"; run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
	dataset="pittsburgh"; dataset_split="test";run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
	dataset="tokyo247"; dataset_split="test";run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
	dataset="tokyoTM"; dataset_split="test";run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
fi

if true ; then
#if false ; then
	ckpt="best"
	pooling="netvlad";
	attention="cbam_attention"
	trained_dataset="pitts30k"  # These are used to specify RESUME weight file
	dataset="pittsburgh"; dataset_split="val"; run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
	dataset="pittsburgh"; dataset_split="test";run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
	dataset="tokyo247"; dataset_split="test";run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
	dataset="tokyoTM"; dataset_split="test";run_test_pooling_existing_attention ${pooling} ${attention} ${trained_dataset}
fi
