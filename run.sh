#!/bin/bash

chmod +x ./config.sh
. ./config.sh

# Updated ResNet with no scale and shift parameters for layer normalisation.
# This prevents overfitting to the training set.
if [ "$VER" == 'resnet-1.1c' ]
then
    python3 main.py --ver               $VER                        \
                    --network           'ResNetV2'                  \
                    --d_model           256                         \
                    --n_blocks          40                          \
                    --d_f               64                          \
                    --k                 3                           \
                    --max_d_rate        16                          \
                    --causal            1                           \
                    --unit_type         "ReLU->LN->W+b"             \
                    --loss_fnc          "BinaryCrossentropy"        \
                    --max_epochs        200                         \
                    --resume_epoch      0                           \
                    --test_epoch        200                         \
                    --mbatch_size       8                           \
                    --inp_tgt_type      'MagXi'                     \
                    --map_type          'DBNormalCDF'               \
                    --sample_size       1000                        \
                    --f_s               16000                       \
                    --T_d               32                          \
                    --T_s               16                          \
                    --min_snr           -10                         \
                    --max_snr           20                          \
                    --snr_inter         1                           \
                    --out_type          $OUT_TYPE                   \
                    --save_model        1                           \
                    --log_iter          0                           \
                    --eval_example      1                           \
                    --gain              $GAIN                       \
                    --train             $TRAIN                      \
                    --infer             $INFER                      \
                    --test              $TEST                       \
                    --gpu               $GPU                        \
                    --set_path          $SET_PATH                   \
                    --data_path         $DATA_PATH                  \
                    --test_x_path       $TEST_X_PATH                \
                    --test_s_path       $TEST_S_PATH                \
                    --test_d_path       $TEST_D_PATH                \
                    --out_path          $OUT_PATH                   \
                    --model_path        $MODEL_PATH
fi

# Updated ResNet with no scale and shift parameters for layer normalisation.
# This prevents overfitting to the training set.
if [ "$VER" == 'resnet-1.1n' ]
then
    python3 main.py --ver               $VER                        \
                    --network           'ResNetV2'                  \
                    --d_model           256                         \
                    --n_blocks          40                          \
                    --d_f               64                          \
                    --k                 3                           \
                    --max_d_rate        16                          \
                    --causal            0                           \
                    --unit_type         "ReLU->LN->W+b"             \
                    --loss_fnc          "BinaryCrossentropy"        \
                    --max_epochs        200                         \
                    --resume_epoch      0                           \
                    --test_epoch        180                         \
                    --mbatch_size       8                           \
                    --inp_tgt_type      'MagXi'                     \
                    --map_type          'DBNormalCDF'               \
                    --sample_size       1000                        \
                    --f_s               16000                       \
                    --T_d               32                          \
                    --T_s               16                          \
                    --min_snr           -10                         \
                    --max_snr           20                          \
                    --snr_inter         1                           \
                    --out_type          $OUT_TYPE                   \
                    --save_model        1                           \
                    --log_iter          0                           \
                    --eval_example      1                           \
                    --gain              $GAIN                       \
                    --train             $TRAIN                      \
                    --infer             $INFER                      \
                    --test              $TEST                       \
                    --gpu               $GPU                        \
                    --set_path          $SET_PATH                   \
                    --data_path         $DATA_PATH                  \
                    --test_x_path       $TEST_X_PATH                \
                    --test_s_path       $TEST_S_PATH                \
                    --test_d_path       $TEST_D_PATH                \
                    --out_path          $OUT_PATH                   \
                    --model_path        $MODEL_PATH
fi

if [ "$VER" == 'rdlnet-1.0n' ]
then
    python3 main.py --ver               $VER                        \
                    --network           'RDLNet'                    \
                    --n_blocks          10                          \
                    --length            7                           \
                    --m_1               64                          \
                    --causal            0                           \
                    --unit_type         "ReLU->scale*LN+center->W+b"\
                    --loss_fnc          "BinaryCrossentropy"        \
                    --max_epochs        200                         \
                    --resume_epoch      0                           \
                    --test_epoch        180                         \
                    --mbatch_size       8                           \
                    --inp_tgt_type      'MagXi'                     \
                    --xi_map_type       'DBNormalCDF'               \
                    --sample_size       1000                        \
                    --f_s               16000                       \
                    --T_d               32                          \
                    --T_s               16                          \
                    --min_snr           -10                         \
                    --max_snr           20                          \
                    --snr_inter         1                           \
                    --out_type          $OUT_TYPE                   \
                    --save_model        1                           \
                    --log_iter          0                           \
                    --eval_example      1                           \
                    --gain              $GAIN                       \
                    --train             $TRAIN                      \
                    --infer             $INFER                      \
                    --test              $TEST                       \
                    --gpu               $GPU                        \
                    --set_path          $SET_PATH                   \
                    --data_path         $DATA_PATH                  \
                    --test_x_path       $TEST_X_PATH                \
                    --test_s_path       $TEST_S_PATH                \
                    --test_d_path       $TEST_D_PATH                \
                    --out_path          $OUT_PATH                   \
                    --model_path        $MODEL_PATH
fi

# if [ "$VER" == 'resnet-1.0c' ]
# then
#     python3 main.py --ver               'resnet-1.0c'   \
#                     --network           'ResNet'        \
#                     --d_model           256             \
#                     --n_blocks          40              \
#                     --d_f               64              \
#                     --k                 3               \
#                     --max_d_rate        16              \
#                     --causal            1               \
#                     --centre            1               \
#                     --scale             1               \
#                     --max_epochs        200             \
#                     --resume_epoch      0               \
#                     --test_epoch        100             \
#                     --mbatch_size       8               \
#                     --sample_size       1000            \
#                     --inp_tgt_type      'MagXi'         \
#                     --f_s               16000           \
#                     --T_d               32              \
#                     --T_s               16              \
#                     --min_snr           -10             \
#                     --max_snr           20              \
#                     --snr_inter         1               \
#                     --out_type          'y'             \
#                     --save_model        1               \
#                     --log_iter          0               \
#                     --eval_example      0               \
#                     --gain              $GAIN           \
#                     --train             $TRAIN          \
#                     --infer             $INFER          \
#                     --test              $TEST           \
#                     --gpu               $GPU            \
#                     --set_path          $SET_PATH       \
#                     --data_path         $DATA_PATH      \
#                     --test_x_path       $TEST_X_PATH    \
#                     --test_s_path       $TEST_S_PATH    \
#                     --out_path          $OUT_PATH       \
#                     --model_path        $MODEL_PATH
# fi

# if [ "$VER" == 'resnet-1.0n' ]
# then
#     python3 main.py --ver               'resnet-1.0n'   \
#                     --network           'ResNet'        \
#                     --d_model           256             \
#                     --n_blocks          40              \
#                     --d_f               64              \
#                     --k                 3               \
#                     --max_d_rate        16              \
#                     --causal            0               \
#                     --max_epochs        200             \
#                     --resume_epoch      0               \
#                     --test_epoch        180             \
#                     --mbatch_size       8               \
#                     --sample_size       1000            \
#                     --f_s               16000           \
#                     --T_d               32              \
#                     --T_s               16              \
#                     --min_snr           -10             \
#                     --max_snr           20              \
#                     --snr_inter         1               \
#                     --out_type          'y'             \
#                     --save_model        1               \
#                     --log_iter          0               \
#                     --eval_example      0               \
#                     --gain              $GAIN           \
#                     --train             $TRAIN          \
#                     --infer             $INFER          \
#                     --test              $TEST           \
#                     --gpu               $GPU            \
#                     --set_path          $SET_PATH       \
#                     --data_path         $DATA_PATH      \
#                     --test_x_path       $TEST_X_PATH    \
#                     --test_s_path       $TEST_S_PATH    \
#                     --out_path          $OUT_PATH       \
#                     --model_path        $MODEL_PATH
# fi
#
# if [ "$VER" == 'reslstm-1.0c' ]
# then
#     python3 main.py --ver               'reslstm-1.0c'  \
#                     --network           'ResLSTM'       \
#                     --d_model           512             \
#                     --n_blocks          5               \
#                     --max_epochs        200             \
#                     --resume_epoch      0               \
#                     --test_epoch        10              \
#                     --mbatch_size       8               \
#                     --sample_size       1000            \
#                     --f_s               16000           \
#                     --T_d               32              \
#                     --T_s               16              \
#                     --min_snr           -10             \
#                     --max_snr           20              \
#                     --snr_inter         1               \
#                     --out_type          'y'             \
#                     --save_model        1               \
#                     --log_iter          0               \
#                     --eval_example      0               \
#                     --gain              $GAIN           \
#                     --train             $TRAIN          \
#                     --infer             $INFER          \
#                     --test              $TEST           \
#                     --gpu               $GPU            \
#                     --set_path          $SET_PATH       \
#                     --data_path         $DATA_PATH      \
#                     --test_x_path       $TEST_X_PATH    \
#                     --test_s_path       $TEST_S_PATH    \
#                     --out_path          $OUT_PATH       \
#                     --model_path        $MODEL_PATH
# fi
