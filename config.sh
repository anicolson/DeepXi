#!/bin/bash

PROJ_DIR='deepxi'
NEGATIVE="-"

set -o noglob

case `hostname` in
"fist")  echo "Running on fist."
    SET_PATH='/mnt/ssd/deep_xi_training_set'
    DATA_PATH='/home/aaron/data/'$PROJ_DIR
    TEST_X_PATH='/home/aaron/mnt/aaron/set/deep_xi_test_set/test_noisy_speech'
    TEST_S_PATH='/home/aaron/mnt/aaron/set/deep_xi_test_set/test_clean_speech'
    TEST_D_PATH='/home/aaron/mnt/aaron/set/deep_xi_test_set/test_noise'
    OUT_PATH='/home/aaron/mnt/aaron_root/mnt/hdd1/out/'$PROJ_DIR
    MODEL_PATH='/home/aaron/model/'$PROJ_DIR
    ;;
"pinky-jnr")  echo "Running on pinky-jnr."
    SET_PATH='/home/aaron/set/deep_xi_training_set'
    DATA_PATH='/home/aaron/mnt/fist/data/'$PROJ_DIR
    TEST_X_PATH='/home/aaron/mnt/aaron/set/deep_xi_test_set/test_noisy_speech'
    TEST_S_PATH='/home/aaron/mnt/aaron/set/deep_xi_test_set/test_clean_speech'
    TEST_D_PATH='/home/aaron/mnt/aaron/set/deep_xi_test_set/test_noise'
    OUT_PATH='/home/aaron/mnt/aaron_root/mnt/hdd1/out/'$PROJ_DIR
    MODEL_PATH='/home/aaron/mnt/fist/model/'$PROJ_DIR
    ;;
"stink")  echo "Running on stink."
    SET_PATH='/mnt/ssd/deep_xi_training_set'
    DATA_PATH='/home/aaron/mnt/fist/data/'$PROJ_DIR
    TEST_X_PATH='/home/aaron/mnt/aaron/set/deep_xi_test_set/test_noisy_speech'
    TEST_S_PATH='/home/aaron/mnt/aaron/set/deep_xi_test_set/test_clean_speech'
    TEST_D_PATH='/home/aaron/mnt/aaron/set/deep_xi_test_set/test_noise'
    OUT_PATH='/home/aaron/mnt/aaron_root/mnt/hdd1/out/'$PROJ_DIR
    MODEL_PATH='/home/aaron/mnt/fist/model/'$PROJ_DIR
    ;;
*) echo "This workstation is not known. Using default paths."
    SET_PATH='set'
    DATA_PATH='data'
    TEST_X_PATH='set/test_noisy_speech'
    TEST_S_PATH='set/test_clean_speech'
    TEST_D_PATH='set/test_noise'
    OUT_PATH='out'
    MODEL_PATH='model'
   ;;
esac

get_free_gpu () {
    NUM_GPU=$( nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | wc -l )
    echo "$NUM_GPU total GPU/s."
    if [ $1 -eq 1  ]
    then
        echo 'Sleeping'
        sleep 1m
    fi
    while true
    do
        for (( gpu=0; gpu<$NUM_GPU; gpu++ ))
        do
            VAR1=$( nvidia-smi -i $gpu --query-gpu=pci.bus_id --format=csv,noheader )
            VAR2=$( nvidia-smi -i $gpu --query-compute-apps=gpu_bus_id --format=csv,noheader | head -n 1)
            if [ "$VAR1" != "$VAR2" ]
            then
                return $gpu
            fi
        done
        echo 'Waiting for free GPU.'
        sleep 1m
    done
}

VER=0
TRAIN=0
INFER=0
TEST=0
OUT_TYPE='y'
GAIN='mmse-lsa'

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    case "$KEY" in
            VER)                 VER=${VALUE} ;;
            GPU)                 GPU=${VALUE} ;;
            TRAIN)               TRAIN=${VALUE} ;;
            INFER)               INFER=${VALUE} ;;
            TEST)                TEST=${VALUE} ;;
            OUT_TYPE)            OUT_TYPE=${VALUE} ;;
            GAIN)                GAIN=${VALUE} ;;
            *)
    esac
done

WAIT=0
if [ -z $GPU ]
then
    get_free_gpu $WAIT
    GPU=$?
fi
