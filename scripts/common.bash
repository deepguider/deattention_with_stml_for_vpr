#!/bin/bash

## Usage:
## source scripts/commmon.bash

source scripts/set_dg_dataset.sh

centroid_dir="/home/ccsmm/dg_git/checkpoints_for_image_retrieval/checkpoints/data/centroids"
#centroid_dir="~/dg_git/checkpoints_for_image_retrieval/checkpoints/data/centroids"

SRC_refer="${centroid_dir}/vgg16_pitts30k_64_desc_cen.hdf5.reference"
SRC_deatt="${centroid_dir}/vgg16_pitts30k_64_desc_cen.hdf5.deattention"
DST="${centroid_dir}/vgg16_pitts30k_64_desc_cen.hdf5"

function set_reference_centroid() {
	SRC=${SRC_refer}
	cp ${SRC} ${DST}
	check_centroid
	#DIFF=`diff -urN $SRC $DST`;echo $DIFF  # No message is OK. If you got differ message, you meet error.
}

function set_deattention_centroid() {
	SRC=${SRC_deatt}
	cp ${SRC} ${DST}
	check_centroid
	#DIFF=`diff -urN $SRC $DST`;echo $DIFF  # No message is OK. If you got differ message, you meet error.
}

function check_centroid() {
	DIFF=`diff -urN $SRC_refer $DST`
	if [ -z "$DIFF" ];then
		echo "Centroid(${DST}) is for Reference."
	fi
	DIFF=`diff -urN $SRC_deatt $DST`
	if [ -z "$DIFF" ];then
		echo "Centroid(${DST}) is for Deattention."
	fi
}

custom_dataset_root="/home/ccsmm/dg_git/dataset/ImageRetrievalDB"
function link_custom_dataset() {
	dataset_parent=$1  # "testset5_daejeon_gungdong_dbNaver_qSmartPhone"
	dataset_dir="${custom_dataset_root}/${dataset_parent}"
	if [ -e netvlad_v100_datasets_dg ]; then
	    rm netvlad_v100_datasets_dg
	fi

	echo "Use $dataset_dir as dataset."
	ln -sf $dataset_dir netvlad_v100_datasets_dg
}

function link_custom_dataset_seoul_dbRobot_qRobot_220418() {
	link_custom_dataset "custom_dataset_seoul_dbRobot_qRobot_220418"
}

function link_custom_dataset_daejeon_dbNaver_qPhone() {
	link_custom_dataset "custom_dataset_daejeon_dbNaver_qPhone"
}

function link_custom_dataset_seoul_dbRobot_qRobot_211014() {
	link_custom_dataset "custom_dataset_seoul_dbRobot_qRobot_211014"
}

function link_custom_dataset_seoul_coex_indoor() {
	link_custom_dataset "custom_dataset_seoul_coex_indoor_dbNaver1stFloor_Fov68_qRobot_220628_A1Back"
}

function link_custom_dataset_seoul_dbNaver_qRobot() {
	link_custom_dataset "custom_dataset_seoul_dbNaver_qRobot"
}

#set_reference_centroid
#set_deattention_centroid
#check_centroid
