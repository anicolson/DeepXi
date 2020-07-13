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

if isunix
	matlab_bin = [matlabroot, '/bin/glnxa64'];
	command = ['sudo mv ', matlab_bin, '/libexpat.so.1 ', matlab_bin, '/libexpat.so.1.NOFIND'];
	system(command);
end

gpu_flag = lower(input('Use a GPU? (y/n):', 's'));
if ismember(gpu_flag, {'y', 'yes'})
    gpu_flag = true;
elseif ismember(gpu_flag, {'n', 'no'})
    gpu_flag = false;
else
    error('Please enter either ''y'' or ''n''.')
end

if gpu_flag
    system('source ~/venv/DeepXi/bin/activate; ./run.sh VER=resnet-1.1n INFER=1 GAIN=mmse-lsa');
else
    system('source ~/venv/DeepXi/bin/activate; ./run.sh VER=resnet-1.1n INFER=1 GAIN=mmse-lsa GPU=9999');
end
