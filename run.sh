#!/bin/bash

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

WAIT=0

if [ -z $2 ]
then
    get_free_gpu $WAIT
    GPU=$?
else
    GPU=$2
fi

if [ "$1" == 'ResLSTM' ]
then
    python3 main.py --ver               'reslstm-1a'    \
                    --network           'ResLSTM'       \
                    --d_model           512             \
                    --n_blocks          5               \
                    --train             1               \
                    --max_epochs        100             \
                    --resume_epoch      0               \
                    --infer             0               \
                    --test_epoch        0               \
                    --mbatch_size       8               \
                    --sample_size       1000            \
                    --f_s               16000           \
                    --T_d               32              \
                    --T_s               16              \
                    --min_snr           -10             \
                    --max_snr           20              \
                    --out_type          'y'             \
                    --gain              'mmse-lsa'      \
                    --save_model        1               \
                    --log_iter          0               \
                    --eval_example      1               \
                    --gpu               $GPU            \
                    --set_path          $SET_PATH       \
                    --data_path         $DATA_PATH      \
                    --test_x_path       $TEST_X_PATH    \
                    --out_path          $OUT_PATH       \
                    --model_path        $MODEL_PATH
fi

if [ "$1" == 'TCN' ]
then
    python3 main.py --ver               'TCN-1a'        \
                    --network           'TCN'           \
                    --d_model           256             \
                    --n_blocks          40              \
                    --d_f               64              \
                    --k                 3               \
                    --max_d_rate        16              \
                    --train             1               \
                    --max_epochs        200             \
                    --resume_epoch      0               \
                    --infer             0               \
                    --test_epoch        0               \
                    --mbatch_size       8               \
                    --sample_size       1000            \
                    --f_s               16000           \
                    --T_d               32              \
                    --T_s               16              \
                    --min_snr           -10             \
                    --max_snr           20              \
                    --out_type          'y'             \
                    --gain              'mmse-lsa'      \
                    --save_model        1               \
                    --log_iter          0               \
                    --eval_example      1               \
                    --gpu               $GPU            \
                    --set_path          $SET_PATH       \
                    --data_path         $DATA_PATH      \
                    --test_x_path       $TEST_X_PATH    \
                    --out_path          $OUT_PATH       \
                    --model_path        $MODEL_PATH
fi
