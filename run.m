clear all; close all; clc;
%% TO PERFORM SPEECH ENHANCEMENT:
%
%% 1. Place the noisy speech files in 'set/test_noisy_speech'.
%
%% 2. Run this script.
%
%% 3. The enhanced speech outputs will be in 'out'.
%
%% Note: the script will finish with 'Inference complete.'.
%

gpu_flag = lower(input('Use a GPU? (y/n):', 's'));
if ismember(gpu_flag, {'y', 'yes'})
    gpu_flag = true;
elseif ismember(gpu_flag, {'n', 'no'})
    gpu_flag = false;
else
    error('Please enter either ''y'' or ''n''.')
end

if gpu_flag
    system('source ~/tf/bin/activate; python3 deepxi.py --infer 1 --out_type y --gain mmse-lsa --gpu ''0'' --epoch 76');
else
    system('source ~/tf/bin/activate; python3 deepxi.py --infer 1 --out_type y --gain mmse-lsa --gpu '''' --epoch 76');
end