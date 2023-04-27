#!/bin/bash

## Usage:
## source scripts/test_function.bash

## Run your virtual environment activation script
source ~/.virtualenvs/dg_venv3.6/bin/activate
source scripts/set_gpu.bash;set_gpu
source scripts/common.bash

#last_cmd_dbg="--nGPU 1 --which_cuda=cuda:0 --batchSize 4 --threads 0 --reuse_cache_for_debug"
#last_cmd_dbg="--nGPU 1 --which_cuda=cuda --batchSize 4 --threads 0 --reuse_cache_for_debug"
last_cmd_dbg="--nGPU 1 --batchSize 1 --cacheBatchSize=4 --threads 0 --reuse_cache_for_debug"

function run_test() {
	mode="test"
	dataset=$1  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
	dataset_split=$2  # "val", "test"
	pooling=$3  # netvlad
	exDesc=$4  # Description of network or experiment.  ex) referpaper_baseline, baseline, deatt, etc.
	run_mode=$5  # debug or any strings including empty string of ""
	otheroption1=$6  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption2=$7  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption3=${8} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption4=${9} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption5=${10} # 
	otheroption6=${11} # 
	otheroption7=${12} # 
	otheroption8=${13} # 
	otheroption9=${14} # 
	otheroption10=${15} # --deattention_version=1 --senet_attention --deatt_category_list human vehicle sky
	otheroption11=${16} # 
	otheroption12=${17} # 
	otheroption13=${18} # 
	otheroption14=${19} # 
	otheroption15=${20} # 
	otheroption16=${21} # 
	otheroption17=${22} # 
	otheroption18=${23} # 
	otheroption19=${24} # 
	otheroption20=${25} # 
	exDesc_misc=""
    echo "==============================================="
	echo "Run test with ${dataset_split} dataset, baseline without deattention"
    echo "==============================================="
    date_str=`date +%Y%m%d_%H%M`
    [ -e result_txt ] || mkdir result_txt
    result_fname="result_txt/${mode}_${dataset_split}_${dataset}_${pooling}_${exDesc}_${exDesc_misc}_date${date_str}.txt"
    echo "Run experiment : $result_fname"
	if [ "${run_mode}" = "debug" ]; then
		last_cmd=${last_cmd_dbg}
	else
		last_cmd="--nGPU ${nGPU} --cacheBatchSize=24 --batchSize 4 --threads 8 > ${result_fname}"
	fi
	cmd="python main.py  --dataset=${dataset} --mode=${mode} --split=${dataset_split} --pooling=${pooling}\
				--dataPath=${CHKDIR}/data --runsPath=${CHKDIR}/runs \
				--resume=${RESUME} --ckpt=${ckpt}\
				${otheroption1}	${otheroption2}	${otheroption3} ${otheroption4} ${otheroption5} ${otheroption6} ${otheroption7} ${otheroption8} ${otheroption9} ${otheroption10} ${otheroption11} ${otheroption12} ${otheroption13} ${otheroption14} ${otheroption15} ${otheroption16} ${otheroption17} ${otheroption18} ${otheroption19} ${otheroption20} ${last_cmd}"
	eval ${cmd}
}

function run_test_deatt() {
	mode="test"
	dataset=$1  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
	dataset_split=$2  # "val", "test"
	pooling=$3  # netvlad
	exDesc=$4  # Description of network or experiment.  ex) referpaper_baseline, baseline, deatt, etc.
	run_mode=$5  # debug or any strings including empty string of ""
	otheroption1=$6  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption2=$7  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption3=$8  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption4=${9} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption5=${10} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption6=${11} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption7=${12} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption8=${13} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption9=${14} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption10=${15} # --deattention_version=1 --senet_attention --deatt_category_list human vehicle sky
	otheroption11=${16} # 
	otheroption12=${17} # 
	otheroption13=${18} # 
	otheroption14=${19} # 
	otheroption15=${20} # 
	otheroption16=${21} # 
	otheroption17=${22} # 
	otheroption18=${23} # 
	otheroption19=${24} # 
	otheroption20=${25} # 
	exDesc_misc=""
    echo "==============================================="
	echo "Run test with ${dataset_split} dataset, baseline without deattention"
    echo "==============================================="
    date_str=`date +%Y%m%d_%H%M`
    [ -e result_txt ] || mkdir result_txt
    result_fname="result_txt/${mode}_${dataset_split}_${dataset}_${pooling}_${exDesc}_${exDesc_misc}_date${date_str}.txt"
    echo "Run experiment : $result_fname"
	if [ "${run_mode}" = "debug" ]; then
		last_cmd=${last_cmd_dbg}
	else
		last_cmd="--nGPU ${nGPU} --cacheBatchSize=24 --batchSize 4 --threads 8  > ${result_fname}"
	fi
	cmd="python main.py  --dataset=${dataset} --mode=${mode} --split=${dataset_split} --pooling=${pooling}\
				--deattention \
				--dataPath=${CHKDIR}/data --runsPath=${CHKDIR}/runs \
				--resume=${RESUME} --ckpt=${ckpt}\
				${otheroption1}	${otheroption2}	${otheroption3} ${otheroption4} ${otheroption5} ${otheroption6} ${otheroption7} ${otheroption8} ${otheroption9} ${otheroption10} ${otheroption11} ${otheroption12} ${otheroption13} ${otheroption14} ${otheroption15} ${otheroption16} ${otheroption17} ${otheroption18} ${otheroption19} ${otheroption20} ${last_cmd}"
	eval ${cmd}
}


function run_test_add_clutter_in_q_of_test() {
	mode="test"
	dataset=$1  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
	dataset_split=$2  # "val", "test"
	pooling=$3  # netvlad
	add_clutter_iteration=$4  # 3
	exDesc=$5  # Description of network or experiment.  ex) referpaper_baseline, baseline, deatt, etc.
	run_mode=$6  # debug or any strings including empty string of ""
	otheroption1=$7  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption2=$8  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption3=$9  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption4=${10} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption5=${11} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption6=${12} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption7=${13} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption8=${14} # 
	otheroption9=${15} # 
	otheroption10=${16} #
	otheroption11=${17} # 
	otheroption12=${18} # 
	otheroption13=${19} # 
	otheroption14=${20} # 
	otheroption15=${21} # 
	otheroption16=${22} # 
	otheroption17=${23} # 
	otheroption18=${24} # 
	otheroption19=${25} # 
	otheroption20=${26} # 
	exDesc_misc="with_add_clutter_in_q_of_test_iteration${add_clutter_iteration}"
    echo "==============================================="
	echo "Run test with ${dataset_split} dataset"
    echo "==============================================="
    date_str=`date +%Y%m%d_%H%M`
    [ -e result_txt ] || mkdir result_txt
    result_fname="result_txt/${mode}_${dataset_split}_${dataset}_${pooling}_${exDesc}_${exDesc_misc}_date${date_str}.txt"
    echo "Run experiment : $result_fname"
	if [ "${run_mode}" = "debug" ]; then
		last_cmd=${last_cmd_dbg}
	else
		last_cmd="--nGPU ${nGPU} --cacheBatchSize=24 --batchSize 4 --threads 8  > ${result_fname}"
	fi
	cmd="python main.py  --dataset=${dataset} --mode=${mode} --split=${dataset_split} --pooling=${pooling}\
				--add_clutter_test_q --add_clutter_iteration ${add_clutter_iteration} \
				--dataPath=${CHKDIR}/data --runsPath=${CHKDIR}/runs \
				--resume=${RESUME} --ckpt=${ckpt}\
				${otheroption1}	${otheroption2}	${otheroption3} ${otheroption4} ${otheroption5} ${otheroption6} ${otheroption7} ${otheroption8} ${otheroption9} ${otheroption10} ${otheroption11} ${otheroption12} ${otheroption13} ${otheroption14} ${otheroption15} ${otheroption16} ${otheroption17} ${otheroption18} ${otheroption19} ${otheroption20} ${last_cmd}"
	eval ${cmd}
}

function run_test_deatt_add_clutter_in_q_of_test() {
	mode="test"
	dataset=$1  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
	dataset_split=$2  # "val", "test"
	pooling=$3  # netvlad
	add_clutter_iteration=$4  # 3
	exDesc=$5  # Description of network or experiment.  ex) referpaper_baseline, baseline, deatt, etc.
	run_mode=$6  # debug or any strings including empty string of ""
	otheroption1=$7  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption2=$8  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption3=$9  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption4=${10} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption5=${11} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption6=${12} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption7=${13} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption8=${14} # 
	otheroption9=${15} # 
	otheroption10=${16} #
	otheroption11=${17} # 
	otheroption12=${18} # 
	otheroption13=${19} # 
	otheroption14=${20} # 
	otheroption15=${21} # 
	otheroption16=${22} # 
	otheroption17=${23} # 
	otheroption18=${24} # 
	otheroption19=${25} # 
	otheroption20=${26} # 
	exDesc_misc="with_add_clutter_in_q_of_test_iteration${add_clutter_iteration}"
    echo "==============================================="
	echo "Run test with ${dataset_split} dataset"
    echo "==============================================="
    date_str=`date +%Y%m%d_%H%M`
    [ -e result_txt ] || mkdir result_txt
    result_fname="result_txt/${mode}_${dataset_split}_${dataset}_${pooling}_${exDesc}_${exDesc_misc}_date${date_str}.txt"
    echo "Run experiment : $result_fname"
	if [ "${run_mode}" = "debug" ]; then
		last_cmd=${last_cmd_dbg}
	else
		last_cmd="--nGPU ${nGPU} --cacheBatchSize=24 --batchSize 4 --threads 8  > ${result_fname}"
	fi
	cmd="python main.py  --dataset=${dataset} --mode=${mode} --split=${dataset_split} --pooling=${pooling}\
				--deattention \
				--add_clutter_test_q --add_clutter_iteration ${add_clutter_iteration} \
				--dataPath=${CHKDIR}/data --runsPath=${CHKDIR}/runs \
				--resume=${RESUME} --ckpt=${ckpt}\
				${otheroption1}	${otheroption2}	${otheroption3} ${otheroption4} ${otheroption5} ${otheroption6} ${otheroption7} ${otheroption8} ${otheroption9} ${otheroption10} ${otheroption11} ${otheroption12} ${otheroption13} ${otheroption14} ${otheroption15} ${otheroption16} ${otheroption17} ${otheroption18} ${otheroption19} ${otheroption20} ${last_cmd}"
	eval ${cmd}
}

function run_test_add_clutter_in_dbq_of_test() {
	mode="test"
	dataset=$1  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
	dataset_split=$2  # "val", "test"
	pooling=$3  # netvlad
	add_clutter_iteration=$4  # 3
	exDesc=$5  # Description of network or experiment.  ex) referpaper_baseline, baseline, deatt, etc.
	run_mode=$6  # debug or any strings including empty string of ""
	otheroption1=$7  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption2=$8  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption3=$9  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption4=${10} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption5=${11} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption6=${12} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption7=${13} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption8=${14} # 
	otheroption9=${15} # 
	otheroption10=${16} #
	otheroption11=${17} # 
	otheroption12=${18} # 
	otheroption13=${19} # 
	otheroption14=${20} # 
	otheroption15=${21} # 
	otheroption16=${22} # 
	otheroption17=${23} # 
	otheroption18=${24} # 
	otheroption19=${25} # 
	otheroption20=${26} # 
	exDesc_misc="with_add_clutter_in_dbq_of_test_iteration${add_clutter_iteration}"
    echo "==============================================="
	echo "Run test with ${dataset_split} dataset"
    echo "==============================================="
    date_str=`date +%Y%m%d_%H%M`
    [ -e result_txt ] || mkdir result_txt
    result_fname="result_txt/${mode}_${dataset_split}_${dataset}_${pooling}_${exDesc}_${exDesc_misc}_date${date_str}.txt"
    echo "Run experiment : $result_fname"
	if [ "${run_mode}" = "debug" ]; then
		last_cmd=${last_cmd_dbg}
	else
		last_cmd="--nGPU ${nGPU} --cacheBatchSize=24 --batchSize 4 --threads 8  > ${result_fname}"
	fi
	cmd="python main.py  --dataset=${dataset} --mode=${mode} --split=${dataset_split} --pooling=${pooling}\
				--add_clutter_test_q --add_clutter_test_db --add_clutter_iteration ${add_clutter_iteration} \
				--dataPath=${CHKDIR}/data --runsPath=${CHKDIR}/runs \
				--resume=${RESUME} --ckpt=${ckpt}\
				${otheroption1}	${otheroption2}	${otheroption3} ${otheroption4} ${otheroption5} ${otheroption6} ${otheroption7} ${otheroption8} ${otheroption9} ${otheroption10} ${otheroption11} ${otheroption12} ${otheroption13} ${otheroption14} ${otheroption15} ${otheroption16} ${otheroption17} ${otheroption18} ${otheroption19} ${otheroption20} ${last_cmd}"
	eval ${cmd}
}

function run_test_deatt_add_clutter_in_dbq_of_test() {
	mode="test"
	dataset=$1  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
	dataset_split=$2  # "val", "test"
	pooling=$3  # netvlad
	add_clutter_iteration=$4  # 3
	exDesc=$5  # Description of network or experiment.  ex) referpaper_baseline, baseline, deatt, etc.
	run_mode=$6  # debug or any strings including empty string of ""
	otheroption1=$7  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption2=$8  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption3=$9  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption4=${10} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption5=${11} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption6=${12} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption7=${13} # other single option : "" , "--arch=vgg16" , "--arch=alexnet" 
	otheroption8=${14} # 
	otheroption9=${15} # 
	otheroption10=${16} #
	otheroption11=${17} # 
	otheroption12=${18} # 
	otheroption13=${19} # 
	otheroption14=${20} # 
	otheroption15=${21} # 
	otheroption16=${22} # 
	otheroption17=${23} # 
	otheroption18=${24} # 
	otheroption19=${25} # 
	otheroption20=${26} # 
	exDesc_misc="with_add_clutter_in_dbq_of_test_iteration${add_clutter_iteration}"
    echo "==============================================="
	echo "Run test with ${dataset_split} dataset"
    echo "==============================================="
    date_str=`date +%Y%m%d_%H%M`
    [ -e result_txt ] || mkdir result_txt
    result_fname="result_txt/${mode}_${dataset_split}_${dataset}_${pooling}_${exDesc}_${exDesc_misc}_date${date_str}.txt"
    echo "Run experiment : $result_fname"
	if [ "${run_mode}" = "debug" ]; then
		last_cmd=${last_cmd_dbg}
	else
		last_cmd="--nGPU ${nGPU} --cacheBatchSize=24 --batchSize 4 --threads 8  > ${result_fname}"
	fi
	cmd="python main.py  --dataset=${dataset} --mode=${mode} --split=${dataset_split} --pooling=${pooling}\
				--deattention \
				--add_clutter_test_q --add_clutter_test_db --add_clutter_iteration ${add_clutter_iteration} \
				--dataPath=${CHKDIR}/data --runsPath=${CHKDIR}/runs \
				--resume=${RESUME} --ckpt=${ckpt}\
				${otheroption1}	${otheroption2}	${otheroption3} ${otheroption4} ${otheroption5} ${otheroption6} ${otheroption7} ${otheroption8} ${otheroption9} ${otheroption10} ${otheroption11} ${otheroption12} ${otheroption13} ${otheroption14} ${otheroption15} ${otheroption16} ${otheroption17} ${otheroption18} ${otheroption19} ${otheroption20} ${last_cmd}"
	eval ${cmd}
}



## ---- Variable begin
export CHKDIR="checkpoints"
export TMPDIR='/mnt/ramdisk/'

#export margin=0.1  # not need in test mode
#export lr=0.001    # not need in test mode

export dataset="pittsburgh"  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
export dataset_split="test"  # test, val, test250k
export pooling="netvlad"
export clutter_iteration=3
export deatt_w=0.0005
export ckpt="best"  # "best", "latest"
## ---- Variable end


## ---- Usage in scipts test.sh : for test of reference baseline
#source scripts/test_function.bash
#dataset="pittsburgh"
#split="val"
#pooling="netvlad"
#clutter_iteration=3
#RESUME="pretrained/vgg16_netvlad_checkpoint"
#run_test ${dataset} ${split} ${pooling} "referpaper_baseline" 
#run_test ${dataset} ${split} ${pooling} "referpaper_baseline" debug
#run_test ${dataset} ${split} ${pooling} "referpaper_baseline" debug "--write_attention_map"
#run_test_add_clutter_in_q_of_test ${dataset} ${split} ${pooling} ${clutter_iteration} "referpaper_baseline"
#run_test_add_clutter_in_dbq_of_test ${dataset} ${split} ${pooling} ${clutter_iteration} "referpaper_baseline"


## ---- Usage in scipts test.sh : for test of deatt with clutter
#source scripts/test_function.bash
#dataset="pittsburgh"
#split="val"
#pooling="netvlad"
#clutter_iteration=3

#RESUME='checkpoints/runs/Apr27_20-59-44_vgg16_netvlad_add_clutter3_in_train_deatt_w0.001';ckpt="latest";deatt_w=0.001
#run_test_deatt ${dataset} ${dataset_split} ${pooling} "" "normal" "--write_attention_map"
#run_test_add_clutter_in_q_of_test ${dataset} ${split} ${pooling} ${clutter_iteration} "trained_with_addClutter_dbq"
#run_test_add_clutter_in_dbq_of_test ${dataset} ${split} ${pooling} ${clutter_iteration} "trained_with_addClutter_dbq"

#RESUME='checkpoints/runs/Apr27_20-57-56_vgg16_netvlad_deatt_w0.001';ckpt="best";deatt_w=0.001
#run_test ${dataset} ${split} ${pooling} "trained_with_normal_dataset"
#run_test_add_clutter_in_q_of_test ${dataset} ${split} ${pooling} ${clutter_iteration} "trained_with_normal_dataset"
#run_test_add_clutter_in_dbq_of_test ${dataset} ${split} ${pooling} ${clutter_iteration} "trained_with_normal_dataset"
