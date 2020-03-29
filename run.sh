z#!/bin/bash

PROJ_DIR='deepxi'

case `hostname` in
"fist")  echo "Running on fist."
	SET_PATH='/mnt/ssd/SE_TRAIN_V2'
	DATA_PATH='/home/aaron/data/'$PROJ_DIR
	TEST_X_PATH='/home/aaron/mnt/aaron/set/deep_xi_test_set/test_noisy_speech'
	OUT_PATH='out/'$PROJ_DIR
	# OUT_PATH='/home/aaron/out/'$PROJ_DIR
	MODEL_PATH='/home/aaron/model/'$PROJ_DIR
	;;
"pinky-jnr")  echo "Running on pinky-jnr."
	SET_PATH='/home/aaron/set/SE_TRAIN_V2'
	DATA_PATH='/home/aaron/data/'$PROJ_DIR
	TEST_X_PATH=''
	OUT_PATH=''
	MODEL_PATH='/home/aaron/mnt/fist/model/'$PROJ_DIR
	;;
"stink")  echo "Running on stink." 
	SET_PATH='/mnt/ssd'
	DATA_PATH='/home/aaron/data/'$PROJ_DIR
	TEST_X_PATH=''
	OUT_PATH=''
	MODEL_PATH='/home/aaron/mnt/fist/model/'$PROJ_DIR
	;;
*) echo "This workstation is not known. Using default paths."
	SET_PATH='set'
	DATA_PATH='data'
	TEST_X_PATH='set/test_noisy_speech'
	OUT_PATH='out'
	MODEL_PATH='model'
   ;;
esac

NUM_GPU=$( nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | wc -l )
echo "$NUM_GPU total GPU/s."

get_free_gpu () {
	if [ $2 -eq 1 ]
	then
		echo 'Sleeping'
	    sleep 1m
	fi
	while true
	do
		for (( gpu=0; gpu<$1; gpu++ ))
		do
			VAR1=$( nvidia-smi -i $gpu --query-gpu=pci.bus_id --format=csv,noheader )
			VAR2=$( nvidia-smi -i $gpu --query-compute-apps=gpu_bus_id --format=csv,noheader )
			if [ "$VAR1" != "$VAR2" ]
			then
				return $gpu
			fi
		done
		echo 'Waiting for free GPU.'
		sleep 1m
	done
}

WAIT=0

if [ -z $1 ]
then
	get_free_gpu $NUM_GPU $WAIT
	GPU=$?
else
	GPU=$1
fi

python3 main.py --ver 			'tcn-1h'		\
				--network		'ResLSTM'		\
				--train 		1 				\
				--max_epochs 	200				\
				--resume_epoch 	0 				\
				--infer 		0 				\
				--epoch 		0 				\
				--mbatch_size 	8 				\
				--sample_size 	1000 			\
				--T_w 			32 				\
				--T_s 			16 				\
				--min_snr 		-10 			\
				--max_snr 		20 				\
				--out_type 		'y' 			\
				--gain 			'mmse-lsa' 		\
				--gpu 			$GPU 			\
				--set_path 		$SET_PATH 		\
				--data_path 	$DATA_PATH 		\
				--test_x_path 	$TEST_X_PATH 	\
				--out_path 		$OUT_PATH 		\
				--model_path 	$MODEL_PATH 	


