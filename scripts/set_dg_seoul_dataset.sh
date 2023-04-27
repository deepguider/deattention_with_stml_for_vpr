#!/bin/bash

######### [User Config begin] #############
User=`whoami`

function set_dataset(){
	sel=$1
	dataset_src=""
	[ -z $sel ]&&sel=1 ## If not given, use 1 as default 

	if [ $sel -eq 2 ];then
		## coex 1st floor only
		## coex including 1st and B1 floors
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_coex_indoor_dbNaverFov68_qRobot_220628_A1Back"
	fi

	if [ $sel -eq 3 ];then
		## coex 1st floor from rosbag file
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_dbRobotGPSModified_qRobot_220628_pathA1_back"
	fi

	if [ $sel -eq 4 ];then
		## coex 1st floor from rosbag file
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_daejeon_119_dbRobot360_qRobotCam_220808"
	fi

	#dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_coex_indoor_dbNaverFov68All_qRobot_220628_A1Back"
	#dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_coex_indoor_dbNaverFov68All_qRobot_220628_A1Back_for_test"
	#dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_coex_indoor_dbNaverFov90Front_qRobot_220628_A1Back"
	#dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_coex_indoor_dbNaverFov90FrontBack_qRobot_220628_A1Back"
	#dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_coex_indoor_dbNaverFov120_qRobot_220628_A1Back"
	#dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_coex_indoor_dbNaverFov360_qRobot_220628_A1Back"
	#dataset_src="/home/${User}/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_dbRobot_qRobot_220628"  # User define
	#dataset_src="/home/${User}/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_coex_indoor_dbNaver_qRobot_220628"  # User define
	#dataset_src="/home/${User}/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_coex_indoor_dbNaver_qRobot_220628_A1Back"  # User define
	#dataset_src="/home/${User}/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_dbRobot_qRobot_220418"

	## If valid $sel is not given, use first dataset as default 
	## Default
	[ -z $dataset_src ]&&sel=1 ## If not given, use 1 as default 
	if [ $sel -eq 1 ];then
		## coex 1st floor only
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_coex_indoor_dbNaver1stFloor_Fov68_qRobot_220628_A1Back"
	fi

	echo "====> Setting Dataset to : ${dataset_src}"
}

######### [User Config end] #############

function link_parent(){
    src=$1
    dst=$2
    [ -e ${dst} ] && rm ${dst}
    ln -sf ${src} ${dst}
	echo " ====> Symbolic Link : ln -sf ${src} ${dst}"
}

function link_dataset(){
	set_dataset $1
	dataset_dst="netvlad_v100_datasets_dg"  # Fixed location
	link_parent ${dataset_src} ${dataset_dst}
}

#link_dataset  # run this top of the each train/test script
