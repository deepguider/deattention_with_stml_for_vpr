#!/bin/bash

######### [User Config begin] #############
User=`whoami`

function set_dataset(){
	sel=$1
	dataset_src=""
	[ -z $sel ]&&sel=1 ## If not given, use 1 as default 
	dataset_map_idx=0

	if [ $sel -eq 2 ];then
		## coex 1st floor only
		## coex including 1st and B1 floors
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_coex_indoor_dbNaverFov68_qRobot_220628_A1Back"
		dataset="dg_seoul"
	fi

	if [ $sel -eq 3 ];then
		## coex 1st floor from rosbag file
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_dbRobotGPSModified_qRobot_220628_pathA1_back"
		dataset="dg_seoul"
	fi

	if [ $sel -eq 4 ];then
		## ETRI to doryong-dong 119 from rosbag file
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_daejeon_119_dbRobot360_qRobotCam_220808"
		dataset="dg_daejeon"
	fi

	if [ $sel -eq 5 ];then
		## bucheon KETI to sports complex ground
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_bucheon_KETI2Sport"
		dataset="dg_bucheon"
		dataset_map_idx=1
	fi

	if [ $sel -eq 6 ];then
		## bucheon KETI to sports complex ground
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_bucheon_Station2Cityhall"
		dataset="dg_bucheon"
		dataset_map_idx=0
	fi

	if [ $sel -eq 7 ];then
		## coex 1st floor from rosbag file, db: 0829 modified, q : 0628 modified
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_seoul_dbRobotGPSModified_qRobot_220829_pathA1_back"
		dataset="dg_seoul"
	fi

	if [ $sel -eq 8 ];then
		## bucheon outdoor, db: 221014 at day, q : 221013 at day
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_bucheon_Sport2KETI_sunset"
		dataset="dg_bucheon"
		dataset_map_idx=1
	fi

	if [ $sel -eq 9 ];then
		## bucheon outdoor from rosbag file, db: 221014 at day, q : 221013 at night
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_bucheon_Sport2KETI_night"
		dataset="dg_bucheon"
		dataset_map_idx=1
	fi

	if [ $sel -eq 10 ];then
		## bucheon outdoor to indoor, with modified GPS, db: 221108 at day, q : same file as db
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_bucheon_KETI2Nonghyup"
		dataset="dg_bucheon"
		dataset_map_idx=1
	fi

	if [ $sel -eq 11 ];then
		## (for train) bucheon outdoor to indoor, with modified GPS, db: 221108 at day, q : same file as db
		dataset_src="/home/ccsmm/dg_git/dataset/ImageRetrievalDB/custom_dataset_bucheon_KETI2Nonghyup_train"
		dataset="dg_bucheon"
		dataset_map_idx=1
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
