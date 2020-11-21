#!/bin/bash

PROJ_DIR='deepxi'
NEGATIVE="-"

set -o noglob

## Use hostname or whoami to set the paths on your workstation.

case `hostname` in
"fist")  echo "Running on `hostname`."
    LOG_PATH='log'
    SET_PATH='/mnt/ssd/deep_xi_dataset'
    DATA_PATH='/home/aaron/data/'$PROJ_DIR
    TEST_X_PATH='/home/aaron/mnt/aaron/set/deep_xi_dataset/test_noisy_speech'
    TEST_S_PATH='/home/aaron/mnt/aaron/set/deep_xi_dataset/test_clean_speech'
    TEST_D_PATH='/home/aaron/mnt/aaron/set/deep_xi_dataset/test_noise'
    OUT_PATH='/home/aaron/mnt/aaron_root/mnt/hdd1/out/'$PROJ_DIR
    MODEL_PATH='/home/aaron/model/'$PROJ_DIR
    ;;
"pinky-jnr")  echo "Running on `hostname`."
    LOG_PATH='log'
    SET_PATH='/home/aaron/set/deep_xi_dataset'
    DATA_PATH='/home/aaron/mnt/fist/data/'$PROJ_DIR
    TEST_X_PATH='/home/aaron/mnt/aaron/set/deep_xi_dataset/test_noisy_speech'
    TEST_S_PATH='/home/aaron/mnt/aaron/set/deep_xi_dataset/test_clean_speech'
    TEST_D_PATH='/home/aaron/mnt/aaron/set/deep_xi_dataset/test_noise'
    OUT_PATH='/home/aaron/mnt/aaron_root/mnt/hdd1/out/'$PROJ_DIR
    MODEL_PATH='/home/aaron/mnt/fist/model/'$PROJ_DIR
    ;;
"LANGTON-RBH")  echo "Running on `hostname`."
    LOG_PATH='log'
    SET_PATH='/mnt/c/Users/nic261/Datasets/deep_xi_dataset'
    DATA_PATH='data'
    TEST_X_PATH='/mnt/c/Users/nic261/Datasets/deep_xi_dataset/test_noisy_speech'
    TEST_S_PATH='/mnt/c/Users/nic261/Datasets/deep_xi_dataset/test_clean_speech'
    TEST_D_PATH='/mnt/c/Users/nic261/Datasets/deep_xi_dataset/test_noise'
    OUT_PATH='/mnt/c/Users/nic261/Outputs/'$PROJ_DIR
    MODEL_PATH='model'
    ;;
*) case `whoami` in
    nic261)  echo Running on a cluster.
      LOG_PATH='log'
      SET_PATH='/scratch1/nic261/Datasets/deep_xi_dataset'
      DATA_PATH='data'
      TEST_X_PATH='/scratch1/nic261/Datasets/deep_xi_dataset/test_noisy_speech'
      TEST_S_PATH='/scratch1/nic261/Datasets/deep_xi_dataset/test_clean_speech'
      TEST_D_PATH='/scratch1/nic261/Datasets/deep_xi_dataset/test_noise'
      OUT_PATH='out'
      MODEL_PATH='model'
    ;;
  *) echo "This workstation is not known."
      LOG_PATH='log'
      SET_PATH='set'
      DATA_PATH='data'
      TEST_X_PATH='set/test_noisy_speech'
      TEST_S_PATH='set/test_clean_speech'
      TEST_D_PATH='set/test_noise'
      OUT_PATH='out'
      MODEL_PATH='model'
    ;;
  esac
  ;;
esac

get_free_gpu () {
  echo "Finding GPU/s..."
  if ! [ -x "$(command -v nvidia-smi)" ];
  then
    echo "nvidia-smi does not exist, using CPU instead."
    GPU=-1
  else
    NUM_GPU=$( nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | wc -l )
    echo "$NUM_GPU total GPU/s."
    while true
    do
      for (( GPU=0; GPU<$NUM_GPU; GPU++ ))
      do
        VAR1=$( nvidia-smi -i $GPU --query-gpu=pci.bus_id --format=csv,noheader )
        VAR2=$( nvidia-smi -i $GPU --query-compute-apps=gpu_bus_id --format=csv,noheader | head -n 1)
        if [ "$VAR1" != "$VAR2" ]
        then
          echo "Using GPU $GPU."
          return
        fi
      done
      echo 'Waiting for free GPU.'
      sleep 1m
    done
  fi
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
