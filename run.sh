#!/bin/bash

PROJ_DIR='deepxi'

NUM_GPU=$( nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | wc -l )

case `hostname` in
"fist")  echo "Running on fist."
	SET_PATH='/mnt/ssd/SE_TRAIN_V2'
	DATA_PATH='/home/aaron/data/'$PROJ_DIR
	TEST_X_PATH='/home/aaron/mnt/aaron/set/SE_TEST'
	OUT_PATH='/home/aaron/out'$PROJ_DIR
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

TRAIN=1
MAX_EPOCHS=200
INFER=0
EPOCH=175
MBATCH_SIZE=10
SAMPLE_SIZE=1000
OUT_TYPE='y'
GAIN='mmse-lsa' # if OUT_TYPE is 'y'.
T_W=32
T_S=16
MIN_SNR=-10
MAX_SNR=20
WAIT=0

if [ -z $2 ]
then
	get_free_gpu $NUM_GPU $WAIT
	GPU=$?
else
	GPU=$2
fi
python3 main.py --ver 'VER_NAME' --train $TRAIN --max_epochs $MAX_EPOCHS --infer $INFER --epoch $EPOCH \
	--gpu $GPU --mbatch_size $MBATCH_SIZE --sample_size $SAMPLE_SIZE --set_path $SET_PATH --data_path $DATA_PATH \
	--T_w $T_W --T_s $T_S --min_snr $MIN_SNR --max_snr $MAX_SNR --test_x_path $TEST_X_PATH \
	--out_path $OUT_PATH --model_path $MODEL_PATH --out_type $OUT_TYPE --gain $GAIN

