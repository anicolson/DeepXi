#!/bin/bash

TRAIN=1
MAX_EPOCHS=175
INFER=0
EPOCH=175
MBATCH_SIZE=10
SAMPLE_SIZE=1000
T_W=32
T_S=16
MIN_SNR=-10
MAX_SNR=20
SET_PATH='set'
DATA_PATH='data'
TEST_X_PATH='set/test_noisy_speech'
OUT_PATH='out'
MODEL_PATH='model'
NUM_GPU=1

get_free_gpu () {
	while true
	do
		for (( i=0; i<$1; i++ ))
		do
			VAR1=$( nvidia-smi -i $i --query-gpu=pci.bus_id --format=csv,noheader )
			VAR2=$( nvidia-smi -i $i --query-compute-apps=gpu_bus_id --format=csv,noheader )
			if [ "$VAR1" != "$VAR2" ]
			then
				return $i
			fi
		done
		sleep 10s
	done
}

get_free_gpu $NUM_GPU
python3 deepxi.py --ver '3e' --train $TRAIN --max_epochs $MAX_EPOCHS --infer $INFER --epoch $EPOCH \
	--gpu $? --mbatch_size $MBATCH_SIZE --sample_size $SAMPLE_SIZE --set_path $SET_PATH --data_path $DATA_PATH \
	--T_w $T_W --T_s $T_S --min_snr $MIN_SNR --max_snr $MAX_SNR --test_x_path $TEST_X_PATH \
	--out_path $OUT_PATH --model_path $MODEL_PATH