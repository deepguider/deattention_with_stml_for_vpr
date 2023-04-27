#!/bin/bash

## Run your virtual environment activation script
source ~/.virtualenvs/dg_venv3.6/bin/activate
source scripts/test_function.bash

## Configuration
run_mode="normal"  # Use all GPU and save result to txt files
#run_mode="debug"  # Use a single GPU (to avoid parallel thread in debug mode) and print out result on the screen, and use saved cache features for quickness
clutter_iteration=3

function run_test_our_sTML(){
	extra_options="--tml2";
	exDesc="sTML${tml2_pn_margin}_only"
    RESUME="pretrained/trained_${trained_dataset}_vgg16_${pooling}pooling_sTML1.4_m0.1";
	run_test ${dataset} ${dataset_split} ${pooling} ${exDesc} ${run_mode} ${extra_options}
#    run_test_add_clutter_in_q_of_test ${dataset} ${dataset_split} ${pooling} ${clutter_iteration} ${exDesc} ${run_mode} ${extra_options} 
#    run_test_add_clutter_in_dbq_of_test ${dataset} ${dataset_split} ${pooling} ${clutter_iteration} ${exDesc} ${run_mode} ${extra_options}
}

function run_test_our_deattention_with_sTML(){
	## Add rerank and deatt_weighted_mask
	extra_options="--deattention --tml2 --senet_attention";
	exDesc="sTML${tml2_pn_margin}_deatt_w${deatt_w}_senet_attention"
    RESUME="pretrained/trained_${trained_dataset}_vgg16_${pooling}pooling_sTML1.4_deatt${deatt_w}_reclustering_senet";
	run_test ${dataset} ${dataset_split} ${pooling} ${exDesc} ${run_mode} ${extra_options}
#    run_test_add_clutter_in_q_of_test ${dataset} ${dataset_split} ${pooling} ${clutter_iteration} ${exDesc} ${run_mode} ${extra_options} 
#    run_test_add_clutter_in_dbq_of_test ${dataset} ${dataset_split} ${pooling} ${clutter_iteration} ${exDesc} ${run_mode} ${extra_options}
}

dataset="pittsburgh"; dataset_split="val";trained_dataset="pitts30k";ckpt="best";pooling="netvlad";deatt_w=0.1
run_test_our_sTML
run_test_our_deattention_with_sTML

dataset="pittsburgh"; dataset_split="test";trained_dataset="pitts30k";ckpt="best";pooling="netvlad";deatt_w=0.1
run_test_our_sTML
run_test_our_deattention_with_sTML

dataset="tokyotm"; dataset_split="test";trained_dataset="pitts30k";ckpt="best";pooling="netvlad";deatt_w=0.1
run_test_our_sTML
run_test_our_deattention_with_sTML

dataset="tokyo247"; dataset_split="test";trained_dataset="pitts30k";ckpt="best";pooling="netvlad";deatt_w=0.1
run_test_our_sTML
run_test_our_deattention_with_sTML
