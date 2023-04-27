#!/bin/bash

## Usage:
## source scripts/set_gpu.bash


function set_gpu() {
	## When you use all 4 gpus
	#export nGPU=4  # Avaiable gpu number. default 4. When all GPU is OK. Do not need to set CUDA_VISIBLE_DEVICES
	export nGPU=`nvidia-smi -L | wc -l`  # 4

	## When you use single gpu
	#export CUDA_DEVICE_ORDER="PCI_BUS_ID"  # This should be runned before setting CUDA_VISIBLE_DEVICES
	#export nGPU=1; export CUDA_VISIBLE_DEVICES=0  # It causes system rebooting because of hw bug on first GPU
	
	## When you use 3 gpus (1,2,3 except 0-th)
	#export CUDA_DEVICE_ORDER="PCI_BUS_ID"  # This should be runned before setting CUDA_VISIBLE_DEVICES
	export nGPU=3;export CUDA_VISIBLE_DEVICES=1,2,3  # If you have some problem to use GPU 0, then remove it from the visible list)

	echo " ${nGPU} GPU[s] will be used."
}
