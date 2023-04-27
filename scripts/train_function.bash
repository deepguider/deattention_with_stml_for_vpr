#!/bin/bash

## Run your virtual environment activation script
source ~/.virtualenvs/dg_venv3.6/bin/activate
source scripts/set_gpu.bash;set_gpu
source scripts/common.bash

#last_cmd_dbg="--nGPU 1 --batchSize 4 --threads 0 --reuse_cache_for_debug"
last_cmd_dbg="--nGPU 1 --batchSize 1 --cacheBatchSize=1 --threads 0 --reuse_cache_for_debug"
deatt_category_list="human vehicle"
#deatt_category_list="human vehicle sky"
nEpochs=30

function run_class_statics() {
	mode="class_statics"  # train, test, cluster, class_statics
    dataset=$1  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
	exDesc=$2  # Description of network or experiment.  ex) referpaper_baseline, baseline, deatt, etc.
	run_mode=$3  # debug or any strings including empty string of ""
	otheroption1=$4  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption2=$5  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption3=$6  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption4=$7  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption5=$8  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption6=$9  # other single option : "" , "--arch=vgg16" , "--write_attention_map" 
	otheroption7=${10}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	exDesc_misc=""
	echo ""
    echo "==============================================="
    echo "Run train with ${dataset_split} dataset"
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
                --margin=${margin} --lr=${lr} --nEpochs=${nEpochs} \
                --dataPath=${CHKDIR}/data --runsPath=${CHKDIR}/runs --ckpt=best\
				${otheroption1}	${otheroption2}	${otheroption3} ${otheroption4}	${otheroption5}	${otheroption6}	${otheroption7}	${last_cmd}"
	eval ${cmd}
}

function run_cluster() {
	mode="cluster"  # train, test, cluster
    dataset=$1  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
	exDesc=$2  # Description of network or experiment.  ex) referpaper_baseline, baseline, deatt, etc.
	run_mode=$3  # debug or any strings including empty string of ""
	otheroption1=$4  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption2=$5  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption3=$6  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption4=$7  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption5=$8  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption6=$9  # other single option : "" , "--arch=vgg16" , "--write_attention_map" 
	otheroption7=${10}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	exDesc_misc=""
	echo ""
    echo "==============================================="
    echo "Run train with ${dataset_split} dataset"
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
                --margin=${margin} --lr=${lr} --nEpochs=${nEpochs} \
                --dataPath=${CHKDIR}/data --runsPath=${CHKDIR}/runs --ckpt=best\
				${otheroption1}	${otheroption2}	${otheroption3} ${otheroption4}	${otheroption5}	${otheroption6}	${otheroption7}	${last_cmd}"
	eval ${cmd}
}

function run_train() {
	mode="train"  # train, test, cluster
    dataset=$1  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
    dataset_split=$2  # "train"
    pooling=$3  # netvlad
	exDesc=$4  # Description of network or experiment.  ex) referpaper_baseline, baseline, deatt, etc.
	run_mode=$5  # debug or any strings including empty string of ""
	otheroption1=$6  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption2=$7  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption3=$8  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption4=$9  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption5=${10}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption6=${11}  # other single option : "" , "--arch=vgg16" , "--write_attention_map" 
	otheroption7=${12}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	exDesc_misc=""
	echo ""
    echo "==============================================="
    echo "Run train with ${dataset_split} dataset"
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
                --margin=${margin} --lr=${lr} --nEpochs=${nEpochs} \
                --dataPath=${CHKDIR}/data --runsPath=${CHKDIR}/runs --ckpt=best\
				${otheroption1}	${otheroption2}	${otheroption3} ${otheroption4}	${otheroption5}	${otheroption6}	${otheroption7}	${last_cmd}"
	eval ${cmd}
}

function run_train_deatt_auto() {
	mode="train"  # train, test, cluster
    dataset=$1  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
    dataset_split=$2  # "train"
    pooling=$3  # netvlad
    deatt_w=$4  # It will not affect anything in test mode except writer's directory name
    add_clutter_iteration=$5  # 3
	exDesc=$6  # Description of network or experiment.  ex) referpaper_baseline, baseline, deatt, etc.
	run_mode=$7  # debug or any strings including empty string of ""
	otheroption1=$8  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption2=$9  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption3=${10}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption4=${11}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption5=${12}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption6=${13}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption7=${14}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption8=${15}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption9=${16}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption10=${17}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption11=${18}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption12=${19}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption13=${20}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption14=${21}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption15=${22}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption16=${23}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	exDesc_misc="deatt_w${deatt_w}_trained_with_normal_train_dataset"
	echo ""
    echo "==============================================="
    echo "Run train with ${dataset_split} dataset"
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
#				--resume pretrained/trained_pitts30k_vgg16_netvladpooling_eTML1.4_deatt0.1_reclustering_deattention_auto \
    cmd="python main.py  --dataset=${dataset} --mode=${mode} --split=${dataset_split} --pooling=${pooling}\
                --deattention_auto --w_deatt_loss ${deatt_w} --deatt_category_list ${deatt_category_list} \
                --margin=${margin} --lr=${lr} --nEpochs=${nEpochs} \
                --dataPath=${CHKDIR}/data --runsPath=${CHKDIR}/runs --ckpt=best\
				${otheroption1}	${otheroption2}	${otheroption3} ${otheroption4}	${otheroption5}	${otheroption6}	${otheroption7}	${otheroption8} ${otheroption9} ${otheroption10} ${otheroption11} ${otheroption12} ${otheroption13} ${otheroption14} ${otheroption15} ${otheroption16} ${last_cmd}"
	eval ${cmd}
}


function run_train_deatt() {
	mode="train"  # train, test, cluster
    dataset=$1  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
    dataset_split=$2  # "train"
    pooling=$3  # netvlad
    deatt_w=$4  # It will not affect anything in test mode except writer's directory name
    add_clutter_iteration=$5  # 3
	exDesc=$6  # Description of network or experiment.  ex) referpaper_baseline, baseline, deatt, etc.
	run_mode=$7  # debug or any strings including empty string of ""
	otheroption1=$8  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption2=$9  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption3=${10}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption4=${11}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption5=${12}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption6=${13}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption7=${14}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption8=${15}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption9=${16}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption10=${17}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption11=${18}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption12=${19}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption13=${20}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption14=${21}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption15=${22}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption16=${23}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	exDesc_misc="deatt_w${deatt_w}_trained_with_normal_train_dataset"
	echo ""
    echo "==============================================="
    echo "Run train with ${dataset_split} dataset"
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
                --deattention --w_deatt_loss ${deatt_w} --deatt_category_list ${deatt_category_list} \
                --margin=${margin} --lr=${lr} --nEpochs=${nEpochs} \
                --dataPath=${CHKDIR}/data --runsPath=${CHKDIR}/runs --ckpt=best\
				${otheroption1}	${otheroption2}	${otheroption3} ${otheroption4}	${otheroption5}	${otheroption6}	${otheroption7}	${otheroption8} ${otheroption9} ${otheroption10} ${otheroption11} ${otheroption12} ${otheroption13} ${otheroption14} ${otheroption15} ${otheroption16} ${last_cmd}"
	eval ${cmd}
}

function run_train_deatt_with_add_clutter_dbq_train_dataset() {
	mode="train"  # train, test, cluster
    dataset=$1  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
    dataset_split=$2  # "train"
    pooling=$3  # netvlad
    deatt_w=$4  # It will not affect anything in test mode except writer's directory name
    add_clutter_iteration=$5  # 3
	exDesc=$6  # Description of network or experiment.  ex) referpaper_baseline, baseline, deatt, etc.
	run_mode=$7  # debug or any strings including empty string of ""
	otheroption1=$8  # other single option : "" , "--arch=vgg16" , "--write_attention_map"
	otheroption2=$9  # other single option : "" , "--write_attention_map" , "--write_attention_map"
	otheroption3=${10}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption4=${11}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption5=${12}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption6=${13}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	otheroption7=${14}  # other single option : "" , "--write_attention_map" , "--write_attention_map" 
	exDesc_misc="deatt_w${deatt_w}_trained_with_add_clutter_dbq_Iteration${add_clutter_iteration}_train_datase"
	echo ""
    echo "==============================================="
    echo "Run train with ${dataset_split} dataset"
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
                --deattention --w_deatt_loss ${deatt_w} --deatt_category_list ${deatt_category_list} \
                --add_clutter_train --add_clutter_iteration ${add_clutter_iteration} \
                --margin=${margin} --lr=${lr} --nEpochs=${nEpochs} \
                --dataPath=${CHKDIR}/data --runsPath=${CHKDIR}/runs --ckpt=best\
				${otheroption1}	${otheroption2}	${otheroption3} ${otheroption4}	${otheroption5}	${otheroption6}	${otheroption7}	${last_cmd}"
	eval ${cmd}
}

## ---- Variable begin
export CHKDIR="checkpoints"
export TMPDIR='/mnt/ramdisk/'

export margin=0.1  # not need in test mode
export lr=0.001    # not need in test mode

export dataset="pittsburgh"  # "pittsburgh", "tokyo247", "tokyoTM", "rparis6k", "roxford5k", "dg_daejeon", "dg_seoul", "dg_bucheon"
export dataset_split="test"  # test, val, test250k
export pooling="netvlad"
export clutter_iteration=3
export deatt_w=0.0005
## ---- Variable end


## ---- Usage in scipts test.sh :
#source scripts/train_function.bash
#dataset="pittsburgh"
#split="val"
#pooling="netvlad"
#clutter_iteration=3
#exDesc="baseline"

#run_train ${dataset} ${dataset_split} ${pooling} "baseline"
#run_train ${dataset} ${dataset_split} ${pooling} "baseline" debug
#run_train_deatt  ${dataset} ${dataset_split} ${pooling} ${deatt_w} ${clutter_iteration} "NA"
#run_train_deatt  ${dataset} ${dataset_split} ${pooling} ${deatt_w} ${clutter_iteration} "NA" debug
#run_train_deatt_with_add_clutter_dbq_train_dataset ${dataset} ${dataset_split} ${pooling} ${deatt_w} ${clutter_iteration} ""
